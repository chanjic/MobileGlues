//
// Created by BZLZHH on 2025/1/28.
//

#include "buffer.h"
#include "ankerl/unordered_dense.h"
#include "texture.h"
#include <cstring>
#include <vector>

template <typename K, typename V>
using unordered_map = ankerl::unordered_dense::map<K, V>;

#define DEBUG 1

GLuint bound_array;
static GLint maxBufferId = 0;
static GLint maxArrayId = 0;

// 使用 vector<bool> 替代 vector<char> 节省内存
static std::vector<GLuint> g_gen_buffers;
static std::vector<bool> g_gen_buffer_exists;
static std::vector<GLuint> g_free_buffer_ids;

static std::vector<GLuint> g_gen_arrays;
static std::vector<bool> g_gen_array_exists;
static std::vector<GLuint> g_free_array_ids;

static std::vector<size_t> g_buffer_datasize;
static std::vector<GLuint> g_element_array_buffer_per_vao;

enum BindingIndex : int {
    BI_ARRAY_BUFFER = 0,
    BI_ATOMIC_COUNTER,
    BI_COPY_READ,
    BI_COPY_WRITE,
    BI_DRAW_INDIRECT,
    BI_DISPATCH_INDIRECT,
    BI_ELEMENT_ARRAY,
    BI_PIXEL_PACK,
    BI_PIXEL_UNPACK,
    BI_SHADER_STORAGE,
    BI_TRANSFORM_FEEDBACK,
    BI_UNIFORM_BUFFER,
    BINDING_COUNT
};

// 使用固定大小的数组替代 std::array
static GLuint g_bound_buffers_arr[BINDING_COUNT] = {0};

struct BufferMapping {
    bool isMapped = false;
    GLbitfield access = 0;
    GLintptr offset = 0;
    GLsizeiptr length = 0;
    void* shadowBuffer = nullptr;
    bool isDirty = false;
    bool persistent = false;
};

// 预分配空间
static unordered_map<GLuint, BufferMapping> g_buffer_mapping;

// 使用查表法加速目标到索引的转换
static const GLenum binding_index_map[] = {
    GL_ARRAY_BUFFER, GL_ATOMIC_COUNTER_BUFFER, GL_COPY_READ_BUFFER, 
    GL_COPY_WRITE_BUFFER, GL_DRAW_INDIRECT_BUFFER, GL_DISPATCH_INDIRECT_BUFFER,
    GL_ELEMENT_ARRAY_BUFFER, GL_PIXEL_PACK_BUFFER, GL_PIXEL_UNPACK_BUFFER,
    GL_SHADER_STORAGE_BUFFER, GL_TRANSFORM_FEEDBACK_BUFFER, GL_UNIFORM_BUFFER
};

// 内联函数并预分配额外空间以减少resize次数
static inline void ensure_buffer_capacity(GLuint id) {
    if (g_gen_buffers.size() <= id) {
        size_t new_size = id + 64; // 预分配额外空间
        g_gen_buffers.resize(new_size, 0);
        g_gen_buffer_exists.resize(new_size, false);
        g_buffer_datasize.resize(new_size, 0);
    }
}

static inline void ensure_array_capacity(GLuint id) {
    if (g_gen_arrays.size() <= id) {
        size_t new_size = id + 64; // 预分配额外空间
        g_gen_arrays.resize(new_size, 0);
        g_gen_array_exists.resize(new_size, false);
        g_element_array_buffer_per_vao.resize(new_size, 0);
    }
}

GLuint gen_buffer() {
    if (!g_free_buffer_ids.empty()) {
        GLuint id = g_free_buffer_ids.back();
        g_free_buffer_ids.pop_back();
        ensure_buffer_capacity(id);
        g_gen_buffers[id] = 0;
        g_gen_buffer_exists[id] = true;
        g_buffer_datasize[id] = 0;
        if (id > (GLuint)maxBufferId) maxBufferId = id;
        return id;
    }
    maxBufferId++;
    ensure_buffer_capacity(maxBufferId);
    g_gen_buffers[maxBufferId] = 0;
    g_gen_buffer_exists[maxBufferId] = true;
    g_buffer_datasize[maxBufferId] = 0;
    return maxBufferId;
}

GLboolean has_buffer(GLuint key) {
    return (key < g_gen_buffer_exists.size()) ? g_gen_buffer_exists[key] : false;
}

void modify_buffer(GLuint key, GLuint value) {
    ensure_buffer_capacity(key);
    g_gen_buffers[key] = value;
    g_gen_buffer_exists[key] = true;
}

void remove_buffer(GLuint key) {
    if (key < g_gen_buffer_exists.size() && g_gen_buffer_exists[key]) {
        g_gen_buffer_exists[key] = false;
        g_gen_buffers[key] = 0;
        g_buffer_datasize[key] = 0;
        g_free_buffer_ids.push_back(key);

        auto it = g_buffer_mapping.find(key);
        if (it != g_buffer_mapping.end()) {
            if (it->second.shadowBuffer) {
                free(it->second.shadowBuffer);
            }
            g_buffer_mapping.erase(it);
        }
    }
}

GLuint find_real_buffer(GLuint key) {
    return (key < g_gen_buffers.size() && g_gen_buffer_exists[key]) ? g_gen_buffers[key] : 0;
}

GLuint get_ibo_by_vao(GLuint vao) {
    return (vao < g_element_array_buffer_per_vao.size()) ? g_element_array_buffer_per_vao[vao] : 0;
}

GLuint find_bound_array() {
    return bound_array;
}

void update_vao_ibo_binding(GLuint vao, GLuint ibo) {
    ensure_array_capacity(vao);
    g_element_array_buffer_per_vao[vao] = ibo;
}

void set_buffer_data_size(GLuint buffer, size_t size) {
    ensure_buffer_capacity(buffer);
    g_buffer_datasize[buffer] = size;
}

size_t get_buffer_data_size(GLuint buffer) {
    return (buffer < g_buffer_datasize.size()) ? g_buffer_datasize[buffer] : 0;
}

static inline int binding_target_to_index(GLenum target) {
    for (int i = 0; i < BINDING_COUNT; ++i) {
        if (binding_index_map[i] == target) {
            return i;
        }
    }
    return -1;
}

void set_bound_buffer_by_target(GLenum target, GLuint buffer) {
    int idx = binding_target_to_index(target);
    if (idx >= 0) g_bound_buffers_arr[idx] = buffer;
}

GLuint find_bound_buffer(GLenum key) {
    // 使用静态映射表避免重复计算
    static const std::pair<GLenum, GLenum> binding_pairs[] = {
        {GL_ARRAY_BUFFER_BINDING, GL_ARRAY_BUFFER},
        {GL_ATOMIC_COUNTER_BUFFER_BINDING, GL_ATOMIC_COUNTER_BUFFER},
        {GL_COPY_READ_BUFFER_BINDING, GL_COPY_READ_BUFFER},
        {GL_COPY_WRITE_BUFFER_BINDING, GL_COPY_WRITE_BUFFER},
        {GL_DRAW_INDIRECT_BUFFER_BINDING, GL_DRAW_INDIRECT_BUFFER},
        {GL_DISPATCH_INDIRECT_BUFFER_BINDING, GL_DISPATCH_INDIRECT_BUFFER},
        {GL_ELEMENT_ARRAY_BUFFER_BINDING, GL_ELEMENT_ARRAY_BUFFER},
        {GL_PIXEL_PACK_BUFFER_BINDING, GL_PIXEL_PACK_BUFFER},
        {GL_PIXEL_UNPACK_BUFFER_BINDING, GL_PIXEL_UNPACK_BUFFER},
        {GL_SHADER_STORAGE_BUFFER_BINDING, GL_SHADER_STORAGE_BUFFER},
        {GL_TRANSFORM_FEEDBACK_BUFFER_BINDING, GL_TRANSFORM_FEEDBACK_BUFFER},
        {GL_UNIFORM_BUFFER_BINDING, GL_UNIFORM_BUFFER}
    };

    if (key == GL_ELEMENT_ARRAY_BUFFER_BINDING) {
        return get_ibo_by_vao(bound_array);
    }

    for (const auto& pair : binding_pairs) {
        if (key == pair.first) {
            int idx = binding_target_to_index(pair.second);
            return (idx >= 0) ? g_bound_buffers_arr[idx] : 0;
        }
    }
    return 0;
}

struct atomic_buffer {
    GLuint id;
    GLsizeiptr size;
    GLintptr offset;
};

static std::vector<atomic_buffer> g_buffer_map_atomic_buffer_info;
static std::vector<GLuint> g_buffer_map_ssbo_id;

void bindAllAtomicCounterAsSSBO() {
    for (size_t i = 0; i < g_buffer_map_atomic_buffer_info.size(); ++i) {
        const atomic_buffer& buf = g_buffer_map_atomic_buffer_info[i];
        if (buf.id != 0) {
            GLuint realID = find_real_buffer(buf.id);
            GLES.glBindBufferRange(GL_SHADER_STORAGE_BUFFER, i, realID, buf.offset, buf.size);
        }
    }
}

// 实现完整的内部格式大小计算
size_t get_internal_format_size(GLenum internalformat) {
    switch (internalformat) {
        case GL_R8: return 1;
        case GL_R8I: case GL_R8UI: return 1;
        case GL_R16: return 2;
        case GL_R16I: case GL_R16UI: case GL_R16F: return 2;
        case GL_R32I: case GL_R32UI: case GL_R32F: return 4;

        case GL_RG8: return 2;
        case GL_RG8I: case GL_RG8UI: return 2;
        case GL_RG16: return 4;
        case GL_RG16I: case GL_RG16UI: case GL_RG16F: return 4;
        case GL_RG32I: case GL_RG32UI: case GL_RG32F: return 8;

        case GL_RGB8: return 3;
        case GL_RGB8I: case GL_RGB8UI: return 3;
        case GL_RGB16: return 6;
        case GL_RGB16I: case GL_RGB16UI: case GL_RGB16F: return 6;
        case GL_RGB32I: case GL_RGB32UI: case GL_RGB32F: return 12;

        case GL_RGBA8: return 4;
        case GL_RGBA8I: case GL_RGBA8UI: return 4;
        case GL_RGBA16: return 8;
        case GL_RGBA16I: case GL_RGBA16UI: case GL_RGBA16F: return 8;
        case GL_RGBA32I: case GL_RGBA32UI: case GL_RGBA32F: return 16;

        case GL_DEPTH_COMPONENT16: return 2;
        case GL_DEPTH_COMPONENT24: return 3;
        case GL_DEPTH_COMPONENT32: return 4;
        case GL_DEPTH_COMPONENT32F: return 4;
        case GL_DEPTH24_STENCIL8: return 4;
        case GL_DEPTH32F_STENCIL8: return 5;

        case GL_STENCIL_INDEX8: return 1;

        case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:
            return 8;
        case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:
            return 16;

        default:
            // 安全默认值：4字节
            return 4;
    }
}
void glGenBuffers(GLsizei n, GLuint* buffers) {
    for (int i = 0; i < n; ++i) {
        buffers[i] = gen_buffer();
    }
}

void glDeleteBuffers(GLsizei n, const GLuint* buffers) {
    for (int i = 0; i < n; ++i) {
        if (find_real_buffer(buffers[i])) {
            GLuint real_buff = find_real_buffer(buffers[i]);
            GLES.glDeleteBuffers(1, &real_buff);
        }
        remove_buffer(buffers[i]);
    }
}

GLboolean glIsBuffer(GLuint buffer) {
    return has_buffer(buffer);
}

void glBindBuffer(GLenum target, GLuint buffer) {
    set_bound_buffer_by_target(target, buffer);
    if (target == GL_ELEMENT_ARRAY_BUFFER) {
        update_vao_ibo_binding(bound_array, buffer);
    }

    if (!has_buffer(buffer) || buffer == 0) {
        GLES.glBindBuffer(target, buffer);
        return;
    }

    GLuint real_buffer = find_real_buffer(buffer);
    if (!real_buffer) {
        GLES.glGenBuffers(1, &real_buffer);
        modify_buffer(buffer, real_buffer);
    }
    GLES.glBindBuffer(target, real_buffer);
}

void glBindBufferRange(GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size) {
    if (!has_buffer(buffer) || buffer == 0) {
        GLES.glBindBufferRange(target, index, buffer, offset, size);
        return;
    }

    GLuint real_buffer = find_real_buffer(buffer);
    if (!real_buffer) {
        GLES.glGenBuffers(1, &real_buffer);
        modify_buffer(buffer, real_buffer);
    }
    GLES.glBindBufferRange(target, index, real_buffer, offset, size);
    if (target == GL_ATOMIC_COUNTER_BUFFER) {
        if (g_buffer_map_atomic_buffer_info.empty()) {
            g_buffer_map_atomic_buffer_info.resize(GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS, {});
        }
        g_buffer_map_atomic_buffer_info[index] = {buffer, size, offset};
    }
}

void glBindBufferBase(GLenum target, GLuint index, GLuint buffer) {
    if (!has_buffer(buffer) || buffer == 0) {
        GLES.glBindBufferBase(target, index, buffer);
        return;
    }

    GLuint real_buffer = find_real_buffer(buffer);
    if (!real_buffer) {
        GLES.glGenBuffers(1, &real_buffer);
        modify_buffer(buffer, real_buffer);
    }
    GLES.glBindBufferBase(target, index, real_buffer);
    if (target == GL_SHADER_STORAGE_BUFFER) {
        if (g_buffer_map_ssbo_id.empty()) {
            g_buffer_map_ssbo_id.resize(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, 0);
        }
        g_buffer_map_ssbo_id[index] = buffer;
    }
}

void glBindVertexBuffer(GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride) {
    if (!has_buffer(buffer) || buffer == 0) {
        GLES.glBindVertexBuffer(bindingindex, buffer, offset, stride);
        return;
    }

    GLuint real_buffer = find_real_buffer(buffer);
    if (!real_buffer) {
        GLES.glGenBuffers(1, &real_buffer);
        modify_buffer(buffer, real_buffer);
    }
    GLES.glBindVertexBuffer(bindingindex, real_buffer, offset, stride);
}

void glTexBuffer(GLenum target, GLenum internalformat, GLuint buffer) {
    if (target != GL_TEXTURE_BUFFER) return;

    if (!has_buffer(buffer) || buffer == 0) {
        GLES.glTexBuffer(target, internalformat, buffer);
        return;
    }

    GLuint real_buffer = find_real_buffer(buffer);
    if (!real_buffer) {
        GLES.glGenBuffers(1, &real_buffer);
        modify_buffer(buffer, real_buffer);
    }

    if (hardware->emulate_texture_buffer) {
        // 优化：简化纹理缓冲区模拟，减少状态保存和恢复次数
        GLint boundTexture = 0;
        GLint prev_pixel_buffer_binding = 0;

        GLES.glActiveTexture(GL_TEXTURE0 + 15);
        GLES.glGetIntegerv(GL_TEXTURE_BINDING_2D, &boundTexture);
        if (!boundTexture) return;

        GLES.glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &prev_pixel_buffer_binding);

        GLES.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, real_buffer);
        GLint bufferSize;
        GLES.glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &bufferSize);
        GLES.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        GLES.glBindTexture(GL_TEXTURE_2D, boundTexture);

        const GLuint MAX_WIDTH = 8192;
        GLuint pixelSize = get_internal_format_size(internalformat);
        GLuint numElements = bufferSize / pixelSize;

        GLuint width = numElements;
        GLuint height = 1;
        if (width > MAX_WIDTH) {
            width = MAX_WIDTH;
            height = (numElements + MAX_WIDTH - 1) / MAX_WIDTH;
        }

        // 保存状态
        GLint prev_alignment, prev_row_length, prev_skip_pixels, prev_skip_rows;
        GLES.glGetIntegerv(GL_UNPACK_ALIGNMENT, &prev_alignment);
        GLES.glGetIntegerv(GL_UNPACK_ROW_LENGTH, &prev_row_length);
        GLES.glGetIntegerv(GL_UNPACK_SKIP_PIXELS, &prev_skip_pixels);
        GLES.glGetIntegerv(GL_UNPACK_SKIP_ROWS, &prev_skip_rows);

        GLES.glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
        GLES.glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);

        GLES.glTexImage2D(GL_TEXTURE_2D, 0, internalformat, width, height, 0, GL_RED_INTEGER, GL_BYTE, nullptr);
        GLES.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, real_buffer);

        // 优化：使用更高效的纹理上传方式（如果高度为1，则一次上传）
        if (height == 1) {
            GLES.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, 1, GL_RED_INTEGER, GL_BYTE, nullptr);
        } else {
            for (GLuint row = 0; row < height; ++row) {
                void* offset = (void*)(row * width * pixelSize);
                GLES.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, row, width, 1, GL_RED_INTEGER, GL_BYTE, offset);
            }
        }

        // 恢复状态
        GLES.glPixelStorei(GL_UNPACK_ALIGNMENT, prev_alignment);
        GLES.glPixelStorei(GL_UNPACK_ROW_LENGTH, prev_row_length);
        GLES.glPixelStorei(GL_UNPACK_SKIP_PIXELS, prev_skip_pixels);
        GLES.glPixelStorei(GL_UNPACK_SKIP_ROWS, prev_skip_rows);

        auto tex = mgGetTexObjectByTarget(target);
        tex->target = ConvertGLEnumToTextureTarget(target);
        tex->internal_format = internalformat;
        tex->width = width;
        tex->height = height;
        tex->depth = 1;
        tex->swizzle_param[0] = GL_RED;
        tex->swizzle_param[1] = GL_GREEN;
        tex->swizzle_param[2] = GL_BLUE;
        tex->swizzle_param[3] = GL_ALPHA;

        GLES.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        GLES.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        GLES.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        GLES.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        GLES.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        GLES.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

        GLES.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, prev_pixel_buffer_binding);
        GLES.glActiveTexture(GL_TEXTURE0 + gl_state->current_tex_unit);
        return;
    }

    GLES.glTexBuffer(target, internalformat, real_buffer);
}

void glTexBufferRange(GLenum target, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size) {
    if (!has_buffer(buffer) || buffer == 0) {
        GLES.glTexBufferRange(target, internalformat, buffer, offset, size);
        return;
    }

    GLuint real_buffer = find_real_buffer(buffer);
    if (!real_buffer) {
        GLES.glGenBuffers(1, &real_buffer);
        modify_buffer(buffer, real_buffer);
    }
    GLES.glTexBufferRange(target, internalformat, real_buffer, offset, size);
}

void glBufferData(GLenum target, GLsizeiptr size, const void* data, GLenum usage) {
    printf("[MGLOG] glBufferData: target=%x, buffer=%u, size=%zd\n", target, buffer, size);
    GLuint buffer = find_bound_buffer(target);
    if (buffer && has_buffer(buffer)) {
        auto& mapping = g_buffer_mapping[buffer];
        // 优化：重用现有内存，避免频繁分配和释放
        if (mapping.shadowBuffer && mapping.length != size) {
            free(mapping.shadowBuffer);
            mapping.shadowBuffer = nullptr;
        }

        if (!mapping.shadowBuffer) {
            mapping.shadowBuffer = malloc(size);
            // 关键修复：确保新分配的缓冲区初始化为0
            if (mapping.shadowBuffer) {
                memset(mapping.shadowBuffer, 0, size);
            }
        }

        if (data && mapping.shadowBuffer) {
            memcpy(mapping.shadowBuffer, data, size);
        }
        // 如果没有提供数据，保持初始化的0值
    }

    GLES.glBufferData(target, size, data, usage);
    set_buffer_data_size(buffer, size);
}

void glBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, const void *data) {
    GLuint buffer = find_bound_buffer(target);
    if (buffer && has_buffer(buffer)) {
        auto it = g_buffer_mapping.find(buffer);
        if (it != g_buffer_mapping.end() && it->second.shadowBuffer) {
            void* shadowPtr = static_cast<char*>(it->second.shadowBuffer) + offset;
            memcpy(shadowPtr, data, size);

            if (it->second.isMapped && (it->second.access & GL_MAP_WRITE_BIT)) {
                it->second.isDirty = true;
            }
        }
    }

    GLES.glBufferSubData(target, offset, size, data);
}

void glGenVertexArrays(GLsizei n, GLuint* arrays) {
    for (int i = 0; i < n; ++i) {
        arrays[i] = gen_array();
    }
}

void glDeleteVertexArrays(GLsizei n, const GLuint* arrays) {
    for (int i = 0; i < n; ++i) {
        if (find_real_array(arrays[i])) {
            GLuint real_array = find_real_array(arrays[i]);
            GLES.glDeleteVertexArrays(1, &real_array);
        }
        remove_array(arrays[i]);
    }
}

GLboolean glIsVertexArray(GLuint array) {
    return has_array(array);
}

void glBindVertexArray(GLuint array) {
    bound_array = array;
    // 更新绑定的IBO
    set_bound_buffer_by_target(GL_ELEMENT_ARRAY_BUFFER, get_ibo_by_vao(array));

    if (!has_array(array) || array == 0) {
        GLES.glBindVertexArray(array);
        return;
    }

    GLuint real_array = find_real_array(array);
    if (!real_array) {
        GLES.glGenVertexArrays(1, &real_array);
        modify_array(array, real_array);
    }
    GLES.glBindVertexArray(real_array);
}
void* glMapBuffer(GLenum target, GLenum access) {
    GLuint buffer = find_bound_buffer(target);
    if (!buffer || !has_buffer(buffer)) {
        return nullptr;
    }

    size_t size = get_buffer_data_size(buffer);
    if (size == 0) {
        return nullptr;
    }

    GLbitfield flags = 0;
    switch (access) {
        case GL_READ_ONLY:
            flags = GL_MAP_READ_BIT;
            break;
        case GL_WRITE_ONLY:
            flags = GL_MAP_WRITE_BIT;
            break;
        case GL_READ_WRITE:
            flags = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT;
            break;
        default:
            return nullptr;
    }

    return glMapBufferRange(target, 0, size, flags);
}

#if GLOBAL_DEBUG || DEBUG
#include <fstream>
#define BIN_FILE_PREFIX "/sdcard/MG/buf/"
#endif

#if !defined(__APPLE__)
extern "C"
{
GLAPI GLAPIENTRY void* glMapBufferARB(GLenum target, GLenum access) __attribute__((alias("glMapBuffer")));
GLAPI GLAPIENTRY void glBufferDataARB(GLenum target, GLsizeiptr size, const void* data, GLenum usage)
__attribute__((alias("glBufferData")));
GLAPI GLAPIENTRY GLboolean glUnmapBufferARB(GLenum target) __attribute__((alias("glUnmapBuffer")));
GLAPI GLAPIENTRY void glBufferStorageARB(GLenum target, GLsizeiptr size, const void* data, GLbitfield flags)
__attribute__((alias("glBufferStorage")));
GLAPI GLAPIENTRY void glBindBufferARB(GLenum target, GLuint buffer) __attribute__((alias("glBindBuffer")));
GLAPI GLAPIENTRY void glBufferSubDataARB(GLenum target, GLintptr offset, GLsizeiptr size, const void *data) __attribute__((alias("glBufferSubData")));
}
#endif

// ================ 关键修复开始 ================ //
void* glMapBufferRange(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access) {
    GLuint buffer = find_bound_buffer(target);
    
    
    // 修复点1: 正确处理非托管缓冲区
    if (!buffer || !has_buffer(buffer) || buffer == 0) {
        return GLES.glMapBufferRange(target, offset, length, access);
    }

    size_t bufferSize = get_buffer_data_size(buffer);
    printf("[MGLOG] glMapBufferRange: target=%x, buffer=%u, offset=%zd, length=%zd, bufferSize=%zd\n", target, buffer, offset, length, bufferSize);
    if (bufferSize == 0) {
        printf("[MGLOG] ERROR: bufferSize==0 for buffer %u!\n", buffer);
    }
    
    if (offset < 0 || (size_t)(offset + length) > bufferSize) {
        return nullptr;
    }

    auto& mapping = g_buffer_mapping[buffer];
    if (mapping.isMapped) {
        return nullptr;
    }

    if (!mapping.shadowBuffer) {
        mapping.shadowBuffer = malloc(bufferSize);
        if (!mapping.shadowBuffer) {
            return nullptr;
        }

        // 关键修复：确保新分配的缓冲区初始化为0
        memset(mapping.shadowBuffer, 0, bufferSize);

        // 修复点2: 移除强制从GPU读取数据的逻辑
        // 仅在映射为写操作时标记为脏数据
        if (access & GL_MAP_WRITE_BIT) {
            mapping.isDirty = true;
        }
    }

    mapping.isMapped = true;
    mapping.access = access;
    mapping.offset = offset;
    mapping.length = length;
    mapping.persistent = (access & GL_MAP_PERSISTENT_BIT) != 0;

    return static_cast<char*>(mapping.shadowBuffer) + offset;
}

GLboolean glUnmapBuffer(GLenum target) {
    GLuint buffer = find_bound_buffer(target);
    
    // 修复点3: 正确处理非托管缓冲区
    if (!buffer || !has_buffer(buffer) || buffer == 0) {
        return GLES.glUnmapBuffer(target);
    }

    auto it = g_buffer_mapping.find(buffer);
    if (it == g_buffer_mapping.end() || !it->second.isMapped) {
        return GL_FALSE;
    }

    auto& mapping = it->second;
    GLboolean result = GL_TRUE;

    // 优化：仅在数据脏且需要写入时才上传到GPU
    if ((mapping.access & GL_MAP_WRITE_BIT) && mapping.isDirty) {
        GLuint real_buffer = find_real_buffer(buffer);
        if (real_buffer) {
            GLES.glBindBuffer(target, real_buffer);
            GLES.glBufferSubData(target, mapping.offset, mapping.length,
                                 static_cast<char*>(mapping.shadowBuffer) + mapping.offset);
            mapping.isDirty = false;
        } else {
            result = GL_FALSE;
        }
    }

    mapping.isMapped = false;
    return result;
}
// ================ 关键修复结束 ================ //

void glBufferStorage(GLenum target, GLsizeiptr size, const void* data, GLbitfield flags) {
    GLenum usage = (flags & GL_DYNAMIC_STORAGE_BIT) ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW;
    GLES.glBufferData(target, size, data, usage);

    GLuint buffer = find_bound_buffer(target);
    if (buffer && has_buffer(buffer)) {
        auto& mapping = g_buffer_mapping[buffer];
        // 优化：重用现有内存
        if (mapping.shadowBuffer && mapping.length != size) {
            free(mapping.shadowBuffer);
            mapping.shadowBuffer = nullptr;
        }

        if (!mapping.shadowBuffer) {
            mapping.shadowBuffer = malloc(size);
            // 关键修复：确保新分配的缓冲区初始化为0
            if (mapping.shadowBuffer) {
                memset(mapping.shadowBuffer, 0, size);
            }
        }

        if (data && mapping.shadowBuffer) {
            memcpy(mapping.shadowBuffer, data, size);
        }
        mapping.isMapped = false;
    }

    set_buffer_data_size(buffer, size);
}

void glFlushMappedBufferRange(GLenum target, GLintptr offset, GLsizeiptr length) {
    GLuint buffer = find_bound_buffer(target);
    
    // 修复点4: 正确处理非托管缓冲区
    if (!buffer || !has_buffer(buffer) || buffer == 0) {
        GLES.glFlushMappedBufferRange(target, offset, length);
        return;
    }

    auto it = g_buffer_mapping.find(buffer);
    if (it == g_buffer_mapping.end() || !it->second.isMapped) {
        return;
    }

    auto& mapping = it->second;
    if (!(mapping.access & GL_MAP_WRITE_BIT)) {
        return;
    }

    if (offset < mapping.offset || offset + length > mapping.offset + mapping.length) {
        return;
    }

    GLuint real_buffer = find_real_buffer(buffer);
    if (real_buffer) {
        GLES.glBindBuffer(target, real_buffer);
        GLES.glBufferSubData(target, mapping.offset + offset, length, 
                             static_cast<char*>(mapping.shadowBuffer) + offset + mapping.offset);
    }
}

void glCopyBufferSubData(GLenum readTarget, GLenum writeTarget, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size) {
    GLuint readBuffer = find_bound_buffer(get_binding_query(readTarget));
    if (!readBuffer || !has_buffer(readBuffer)) {
        return;
    }

    GLuint writeBuffer = find_bound_buffer(get_binding_query(writeTarget));
    if (!writeBuffer || !has_buffer(writeBuffer)) {
        return;
    }

    // 优化：直接使用OpenGL的复制功能，避免映射操作
    GLuint realReadBuffer = find_real_buffer(readBuffer);
    GLuint realWriteBuffer = find_real_buffer(writeBuffer);

    if (realReadBuffer && realWriteBuffer) {
        GLES.glBindBuffer(readTarget, realReadBuffer);
        GLES.glBindBuffer(writeTarget, realWriteBuffer);
        GLES.glCopyBufferSubData(readTarget, writeTarget, readOffset, writeOffset, size);
    }

    // 更新影子缓冲区
    auto itRead = g_buffer_mapping.find(readBuffer);
    auto itWrite = g_buffer_mapping.find(writeBuffer);

    if (itRead != g_buffer_mapping.end() && itRead->second.shadowBuffer &&
        itWrite != g_buffer_mapping.end() && itWrite->second.shadowBuffer) {
        void* srcPtr = static_cast<char*>(itRead->second.shadowBuffer) + readOffset;
        void* dstPtr = static_cast<char*>(itWrite->second.shadowBuffer) + writeOffset;
        memcpy(dstPtr, srcPtr, size);

        if (itWrite->second.isMapped && (itWrite->second.access & GL_MAP_WRITE_BIT)) {
            itWrite->second.isDirty = true;
        }
    } else if (itWrite != g_buffer_mapping.end() && itWrite->second.shadowBuffer) {
        free(itWrite->second.shadowBuffer);
        itWrite->second.shadowBuffer = nullptr;
        itWrite->second.isMapped = false;
    }
}

// 初始化函数添加预分配
void InitBufferMap(size_t expectedSize) {
    size_t new_size = expectedSize + 64;
    g_gen_buffers.reserve(new_size);
    g_gen_buffer_exists.reserve(new_size);
    g_buffer_datasize.reserve(new_size);
    g_buffer_mapping.reserve(expectedSize);

    // 初始化为空，不设置大小为1
    g_gen_buffers.clear();
    g_gen_buffer_exists.clear();
    g_buffer_datasize.clear();
}

void InitVertexArrayMap(size_t expectedSize) {
    size_t new_size = expectedSize + 64;
    g_gen_arrays.reserve(new_size);
    g_gen_array_exists.reserve(new_size);
    g_element_array_buffer_per_vao.reserve(new_size);

    // 初始化为空，不设置大小为1
    g_gen_arrays.clear();
    g_gen_array_exists.clear();
    g_element_array_buffer_per_vao.clear();
}

GLenum get_binding_query(GLenum target) {
    switch (target) {
        case GL_ARRAY_BUFFER:
            return GL_ARRAY_BUFFER_BINDING;
        case GL_ELEMENT_ARRAY_BUFFER:
            return GL_ELEMENT_ARRAY_BUFFER_BINDING;
        case GL_PIXEL_PACK_BUFFER:
            return GL_PIXEL_PACK_BUFFER_BINDING;
        case GL_PIXEL_UNPACK_BUFFER:
            return GL_PIXEL_UNPACK_BUFFER_BINDING;
        case GL_COPY_WRITE_BUFFER:
            return GL_COPY_WRITE_BUFFER_BINDING;
        case GL_COPY_READ_BUFFER:
            return GL_COPY_READ_BUFFER_BINDING;
        case GL_UNIFORM_BUFFER:
            return GL_UNIFORM_BUFFER_BINDING;
        case GL_SHADER_STORAGE_BUFFER:
            return GL_SHADER_STORAGE_BUFFER_BINDING;
        case GL_TRANSFORM_FEEDBACK_BUFFER:
            return GL_TRANSFORM_FEEDBACK_BUFFER_BINDING;
        case GL_ATOMIC_COUNTER_BUFFER:
            return GL_ATOMIC_COUNTER_BUFFER_BINDING;
        case GL_DRAW_INDIRECT_BUFFER:
            return GL_DRAW_INDIRECT_BUFFER_BINDING;
        case GL_DISPATCH_INDIRECT_BUFFER:
            return GL_DISPATCH_INDIRECT_BUFFER_BINDING;
        default:
            return 0;
    }
}

GLuint gen_array() {
    if (!g_free_array_ids.empty()) {
        GLuint id = g_free_array_ids.back();
        g_free_array_ids.pop_back();
        ensure_array_capacity(id);
        g_gen_arrays[id] = 0;
        g_gen_array_exists[id] = true;
        g_element_array_buffer_per_vao[id] = 0;
        if (id > (GLuint)maxArrayId) maxArrayId = id;
        return id;
    }
    maxArrayId++;
    ensure_array_capacity(maxArrayId);
    g_gen_arrays[maxArrayId] = 0;
    g_gen_array_exists[maxArrayId] = true;
    g_element_array_buffer_per_vao[maxArrayId] = 0;
    return maxArrayId;
}

GLboolean has_array(GLuint key) {
    return (key < g_gen_array_exists.size()) ? g_gen_array_exists[key] : false;
}

void modify_array(GLuint key, GLuint value) {
    ensure_array_capacity(key);
    g_gen_arrays[key] = value;
    g_gen_array_exists[key] = true;
}

void remove_array(GLuint key) {
    if (key < g_gen_array_exists.size() && g_gen_array_exists[key]) {
        g_gen_array_exists[key] = false;
        g_gen_arrays[key] = 0;
        g_element_array_buffer_per_vao[key] = 0;
        g_free_array_ids.push_back(key);
    }
}

GLuint find_real_array(GLuint key) {
    return (key < g_gen_arrays.size() && g_gen_array_exists[key]) ? g_gen_arrays[key] : 0;
}
