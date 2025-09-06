//
// Created by BZLZHH on 2025/1/29.
//

#include "drawing.h"
#include "buffer.h"
#include "framebuffer.h"
#include "mg.h"
#include "texture.h"
#include <ankerl/unordered_dense.h>
#include <vector>
#include <memory>

#define DEBUG 0

template <typename Key, typename Value>
using unordered_map = ankerl::unordered_dense::map<Key, Value>;

GLuint bufSampelerProg;
GLuint bufSampelerLoc;
std::string bufSampelerName;

extern std::unordered_map<GLuint, bool> program_map_is_sampler_buffer_emulated;
extern std::unordered_map<GLuint, bool> program_map_is_atomic_counter_emulated;

unordered_map<GLuint, SamplerInfo> g_samplerCacheForSamplerBuffer;

// 优化：uniform location 查询缓存
struct UniformLocationCache {
    unordered_map<GLuint, std::vector<GLint>> samplerLocations;
    unordered_map<GLuint, GLint> widthLocations;
    unordered_map<GLuint, GLint> heightLocations;
};
static UniformLocationCache g_uniformLocCache;

void setupBufferTextureUniforms(GLuint program) {
    LOG_D("setupBufferTextureUniforms, program: %d", program);

    if (!program_map_is_sampler_buffer_emulated[program]) return;

    // 使用缓存机制
    auto& progSamplerInfo = g_samplerCacheForSamplerBuffer[program];
    auto& samplerLocVec = g_uniformLocCache.samplerLocations[program];
    auto& locWidth = g_uniformLocCache.widthLocations[program];
    auto& locHeight = g_uniformLocCache.heightLocations[program];

    if (samplerLocVec.empty() || locWidth == 0 || locHeight == 0) {
        GLint locW = GLES.glGetUniformLocation(program, "u_BufferTexWidth");
        GLint locH = GLES.glGetUniformLocation(program, "u_BufferTexHeight");
        if (locW == -1) {
            LOG_W("u_BufferTexWidth uniform not found in program %d", program);
            return;
        }
        locWidth = locW;
        locHeight = locH;

        samplerLocVec.clear();

        GLint numUniforms = 0;
        GLES.glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &numUniforms);
        LOG_D("Program %d has %d active uniforms", program, numUniforms);

        for (GLint i = 0; i < numUniforms; ++i) {
            const GLsizei bufSize = 256;
            GLchar name[bufSize];
            GLsizei length = 0;
            GLint size = 0;
            GLenum type = 0;
            GLES.glGetActiveUniform(program, i, bufSize, &length, &size, &type, name);

            if (type == GL_SAMPLER_2D || type == GL_INT_SAMPLER_2D) {
                GLint locSampler = GLES.glGetUniformLocation(program, name);
                if (locSampler >= 0)
                    samplerLocVec.push_back(locSampler);
            }
        }
    }

    // 使用缓存后的数据
    for (auto locSampler : samplerLocVec) {
        if (locSampler < 0) {
            continue;
        }

        GLuint prev_unit = gl_state->current_tex_unit;
        constexpr GLint unit = 15;

        GLES.glActiveTexture(GL_TEXTURE0 + unit);
        GLint texId = 0;
        GLES.glGetIntegerv(GL_TEXTURE_BINDING_2D, &texId);
        if (texId == 0) {
            GLES.glActiveTexture(GL_TEXTURE0 + prev_unit);
            continue;
        }

        auto texObject = mgGetTexObjectByID(texId);

        GLES.glUniform1i(locSampler, unit);
        GLES.glUniform1i(locWidth, texObject->width);
        GLES.glUniform1i(locHeight, texObject->height);

        GLES.glActiveTexture(GL_TEXTURE0 + prev_unit);
    }
}

void prepareForDraw() {
    LOG_D("prepareForDraw...")
    if (hardware->emulate_texture_buffer) {
        setupBufferTextureUniforms(gl_state->current_program);
    }
}

// 优化：线程局部临时索引缓冲区，减少频繁 malloc/free
static thread_local std::unique_ptr<std::vector<uint8_t>> g_tempIndexBuffer;

void glDrawElementsInstanced(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei primcount) {
    LOG()
    LOG_D("glDrawElementsInstanced, mode: %d, count: %d, type: %d, indices: %p, primcount: %d", mode, count, type,
          indices, primcount)
    prepareForDraw();
    GLES.glDrawElementsInstanced(mode, count, type, indices, primcount);
    CHECK_GL_ERROR
}

void glDrawElements(GLenum mode, GLsizei count, GLenum type, const void* indices) {
    LOG()
    LOG_D("glDrawElements, mode: %d, count: %d, type: %d, indices: %p", mode, count, type, indices)
    prepareForDraw();
    GLES.glDrawElements(mode, count, type, indices);
    CHECK_GL_ERROR
}

void glBindImageTexture(GLuint unit, GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum access,
                        GLenum format) {
    LOG()
    LOG_D("glBindImageTexture, unit: %d, texture: %d, level: %d, layered: %d, layer: %d, access: %d, format: %d", unit,
          texture, level, layered, layer, access, format)
    GLES.glBindImageTexture(unit, texture, level, layered, layer, access, format);
    CHECK_GL_ERROR
}

void glUniform1i(GLint location, GLint v0) {
    LOG()
    LOG_D("glUniform1i, location: %d, v0: %d", location, v0)
    GLES.glUniform1i(location, v0);
    CHECK_GL_ERROR
}

void bindAllAtomicCounterAsSSBO();
void glDispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z) {
    LOG()
    LOG_D("glDispatchCompute, num_groups_x: %d, num_groups_y: %d, num_groups_z: %d", num_groups_x, num_groups_y,
          num_groups_z)
    if (program_map_is_atomic_counter_emulated[gl_state->current_program]) {
        bindAllAtomicCounterAsSSBO();
        LOG_D("Atomic counters bound as SSBOs for program %d", gl_state->current_program);
    } else {
        LOG_D("No atomic counters bound as SSBOs for program %d", gl_state->current_program);
    }
    GLES.glDispatchCompute(num_groups_x, num_groups_y, num_groups_z);
    CHECK_GL_ERROR
}

void glMemoryBarrier(GLbitfield barriers) {
    LOG()
    LOG_D("glMemoryBarrier, barriers: %d", barriers)
    if (program_map_is_atomic_counter_emulated[gl_state->current_program]) {
        barriers |= GL_ATOMIC_COUNTER_BARRIER_BIT;
        barriers |= GL_SHADER_STORAGE_BARRIER_BIT;
    }
    GLES.glMemoryBarrier(barriers);
    CHECK_GL_ERROR
}

void glDrawElementsBaseVertex(GLenum mode, GLsizei count, GLenum type, const void* indices, GLint basevertex) {
    LOG()
    LOG_D("glDrawElementsBaseVertex, mode: %d, count: %d, type: %d, indices: %p, basevertex: %d", mode, count, type,
          indices, basevertex);
    prepareForDraw();
    if (hardware->es_version < 320 && !g_gles_caps.GL_EXT_draw_elements_base_vertex &&
        !g_gles_caps.GL_OES_draw_elements_base_vertex) {
        // TODO: use indirect drawing for GLES 3.1
        LOG_D("Emulating glDrawElementsBaseVertex")
        GLint prevElementBuffer;
        GLES.glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &prevElementBuffer);

        if (basevertex == 0) {
            GLES.glDrawElements(mode, count, type, indices);
            return;
        }

        size_t indexSize;
        switch (type) {
        case GL_UNSIGNED_INT:
            indexSize = sizeof(GLuint);
            break;
        case GL_UNSIGNED_SHORT:
            indexSize = sizeof(GLushort);
            break;
        case GL_UNSIGNED_BYTE:
            indexSize = sizeof(GLubyte);
            break;
        default:
            return;
        }

        // 优化：使用线程局部缓冲区
        if (!g_tempIndexBuffer)
            g_tempIndexBuffer = std::make_unique<std::vector<uint8_t>>();
        g_tempIndexBuffer->resize(count * indexSize);

        void* tempIndices = g_tempIndexBuffer->data();

        if (prevElementBuffer != 0) {
            GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, prevElementBuffer);
            void* srcData =
                GLES.glMapBufferRange(GL_ELEMENT_ARRAY_BUFFER, (GLintptr)indices, count * indexSize, GL_MAP_READ_BIT);

            if (srcData) {
                memcpy(tempIndices, srcData, count * indexSize);
                GLES.glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
            } else {
                return;
            }
        } else {
            memcpy(tempIndices, indices, count * indexSize);
        }

        // 优化：用指针和类型推断减少分支
        switch (type) {
        case GL_UNSIGNED_INT: {
            auto p = reinterpret_cast<GLuint*>(tempIndices);
            for (int j = 0; j < count; ++j) p[j] += basevertex;
            break;
        }
        case GL_UNSIGNED_SHORT: {
            auto p = reinterpret_cast<GLushort*>(tempIndices);
            for (int j = 0; j < count; ++j) p[j] += basevertex;
            break;
        }
        case GL_UNSIGNED_BYTE: {
            auto p = reinterpret_cast<GLubyte*>(tempIndices);
            for (int j = 0; j < count; ++j) p[j] += basevertex;
            break;
        }
        }

        GLuint tempBuffer;
        GLES.glGenBuffers(1, &tempBuffer);
        GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tempBuffer);
        GLES.glBufferData(GL_ELEMENT_ARRAY_BUFFER, count * indexSize, tempIndices, GL_STREAM_DRAW);

        GLES.glDrawElements(mode, count, type, 0);

        GLES.glDeleteBuffers(1, &tempBuffer);
        GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, prevElementBuffer);

        CHECK_GL_ERROR
    } else {
        GLES.glDrawElementsBaseVertex(mode, count, type, indices, basevertex);
    }
    CHECK_GL_ERROR
}
