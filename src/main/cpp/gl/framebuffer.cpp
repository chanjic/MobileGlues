//
// Created by hanji on 2025/2/6.
//

#include "framebuffer.h"
#include "log.h"
#include "../config/settings.h"
#include "FSR1/FSR1.h"
#include <vector>
#include <memory>

#define DEBUG 0

struct attachment_t {
    GLenum textarget = 0;
    GLuint texture = 0;
    GLint level = 0;
};

struct framebuffer_t {
    bool initialized = false;
    std::unique_ptr<attachment_t[]> color_attachments;
    attachment_t depth_attachment = {};
    attachment_t stencil_attachment = {};
};

static GLint MAX_COLOR_ATTACHMENTS = 0;
static GLint MAX_DRAW_BUFFERS = 0;
static GLuint current_draw_fbo = 0;
static GLuint current_read_fbo = 0;
static std::vector<framebuffer_t> framebuffers;

// 内联，减少函数调用损耗
inline void ensure_max_attachments() {
    if (MAX_COLOR_ATTACHMENTS == 0) {
        GLES.glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &MAX_COLOR_ATTACHMENTS);
        if (MAX_COLOR_ATTACHMENTS <= 0) MAX_COLOR_ATTACHMENTS = 8;
    }
    if (MAX_DRAW_BUFFERS == 0) {
        GLES.glGetIntegerv(GL_MAX_DRAW_BUFFERS, &MAX_DRAW_BUFFERS);
        if (MAX_DRAW_BUFFERS <= 0) MAX_DRAW_BUFFERS = 8;
    }
}

// 内联，减少resize频率
inline framebuffer_t& get_framebuffer(GLuint id) {
    if (id >= framebuffers.size()) {
        framebuffers.resize(id + 32); // 增加扩容步长
    }
    return framebuffers[id];
}

void InitFramebufferMap(size_t expectedSize) {
    framebuffers.reserve(expectedSize + 32); // 提前多预留
}

// 用智能指针管理，避免泄漏，且只分配一次
inline void init_framebuffer(framebuffer_t& fbo) {
    if (!fbo.initialized) {
        ensure_max_attachments();
        fbo.color_attachments = std::make_unique<attachment_t[]>(MAX_COLOR_ATTACHMENTS);
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; ++i) {
            fbo.color_attachments[i] = {};
        }
        fbo.initialized = true;
    }
}

void glBindFramebuffer(GLenum target, GLuint framebuffer) {
    ensure_max_attachments();
    framebuffer_t& fbo = get_framebuffer(framebuffer);

    if (framebuffer != 0) {
        init_framebuffer(fbo);
    }
    if (target == GL_DRAW_FRAMEBUFFER || target == GL_FRAMEBUFFER) {
        current_draw_fbo = framebuffer;
    }
    if (target == GL_READ_FRAMEBUFFER || target == GL_FRAMEBUFFER) {
        current_read_fbo = framebuffer;
    }
    GLES.glBindFramebuffer(target, framebuffer);
}

// attachment_t 赋值直接用列表初始化
inline void update_attachment(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level) {
    GLuint current_fbo = (target == GL_READ_FRAMEBUFFER) ? current_read_fbo : current_draw_fbo;
    if (current_fbo == 0) return;
    framebuffer_t& fbo = framebuffers[current_fbo];
    if (attachment >= GL_COLOR_ATTACHMENT0 && attachment < GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS) {
        int index = attachment - GL_COLOR_ATTACHMENT0;
        fbo.color_attachments[index] = {textarget, texture, level};
    } else if (attachment == GL_DEPTH_ATTACHMENT) {
        fbo.depth_attachment = {textarget, texture, level};
    } else if (attachment == GL_STENCIL_ATTACHMENT) {
        fbo.stencil_attachment = {textarget, texture, level};
    }
}

void glFramebufferTexture2D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level) {
    update_attachment(target, attachment, textarget, texture, level);
    GLES.glFramebufferTexture2D(target, attachment, textarget, texture, level);
}

void glFramebufferTexture(GLenum target, GLenum attachment, GLuint texture, GLint level) {
    update_attachment(target, attachment, GL_TEXTURE_2D, texture, level);
    GLES.glFramebufferTexture(target, attachment, texture, level);
}

void glDrawBuffer(GLenum buffer) {
    GLint currentFBO;
    GLES.glGetIntegerv(GL_FRAMEBUFFER_BINDING, &currentFBO);
    if (currentFBO == 0) {
        GLES.glDrawBuffers(1, &buffer);
    } else {
        GLint maxAttachments;
        GLES.glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &maxAttachments);
        if (buffer == GL_NONE) {
            std::vector<GLenum> buffers(maxAttachments, GL_NONE);
            GLES.glDrawBuffers(maxAttachments, buffers.data());
        } else if (buffer >= GL_COLOR_ATTACHMENT0 && buffer < GL_COLOR_ATTACHMENT0 + maxAttachments) {
            std::vector<GLenum> buffers(maxAttachments, GL_NONE);
            buffers[buffer - GL_COLOR_ATTACHMENT0] = buffer;
            GLES.glDrawBuffers(maxAttachments, buffers.data());
        }
    }
}

void glDrawBuffers(GLsizei n, const GLenum* bufs) {
    if (current_draw_fbo == 0) {
        GLES.glDrawBuffers(n, bufs);
        return;
    }
    std::vector<GLenum> new_bufs(n, GL_NONE);
    framebuffer_t& fbo = framebuffers[current_draw_fbo];
    for (int i = 0; i < n; ++i) {
        if (bufs[i] >= GL_COLOR_ATTACHMENT0 && bufs[i] < GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS) {
            GLenum logical_attachment = bufs[i];
            GLenum physical_attachment = GL_COLOR_ATTACHMENT0 + i;
            new_bufs[i] = physical_attachment;
            int index = logical_attachment - GL_COLOR_ATTACHMENT0;
            attachment_t& attach = fbo.color_attachments[index];
            GLES.glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, physical_attachment, attach.textarget, attach.texture, attach.level);
        } else {
            new_bufs[i] = bufs[i];
        }
    }
    GLES.glDrawBuffers(n, new_bufs.data());
}

void glReadBuffer(GLenum src) {
    if (current_read_fbo != 0 && src >= GL_COLOR_ATTACHMENT0 &&
        src < GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS) {
        framebuffer_t& fbo = framebuffers[current_read_fbo];
        int index = src - GL_COLOR_ATTACHMENT0;
        attachment_t& attach = fbo.color_attachments[index];
        GLES.glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, attach.textarget, attach.texture, attach.level);
        GLES.glReadBuffer(GL_COLOR_ATTACHMENT0);
    } else {
        GLES.glReadBuffer(src);
    }
}

GLenum glCheckFramebufferStatus(GLenum target) {
    GLenum status = GLES.glCheckFramebufferStatus(target);
    if (global_settings.ignore_error == IgnoreErrorLevel::Full && status != GL_FRAMEBUFFER_COMPLETE) {
        return GL_FRAMEBUFFER_COMPLETE;
    }
    return status;
}
