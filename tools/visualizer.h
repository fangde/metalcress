/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2025, Yafei Ou and Mahdi Tavakoli
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CR_TEST_VISUALIZER_H
#define CR_TEST_VISUALIZER_H

#include <iostream>
#include <string>
#include <vector>
#include <glad.h>
#include <glfw3.h>
#include <Eigen/Dense>

// A general-purpose window and renderer class.
class Visualizer
{
public:
    // Constructor opens a window with the given width, height, and title.
    Visualizer(int width = 800, int height = 600, const std::string &title = "Visualizer");

    // Destructor cleans up all allocated resources.
    ~Visualizer();

    // Returns true if the window is still open.
    bool running() const;

    void beginRender();

    void endRender();

    void updateParticleData(const std::vector<float> &positions);

    void drawMeshWireframe(const std::vector<float> &vertices,
                           const std::vector<unsigned int> &indices,
                           const Eigen::Vector3f &position,
                           const Eigen::Matrix3f &rotation,
                           const Eigen::Vector3f &scale,
                           const Eigen::Vector4f &color = Eigen::Vector4f(1.0f, 1.0f, 0.0f, 1.0f));

    void drawParticles(const std::vector<float> &positions,
                       const Eigen::Vector4f &color = Eigen::Vector4f(0.5f, 0.5f, 1.0f, 1.0f));

    // Retrieve the GLFW window pointer (if needed for other operations)
    GLFWwindow *getWindow() const { return mWindow; }

private:
    // Window attributes
    int mWidth;
    int mHeight;
    std::string mTitle;
    GLFWwindow *mWindow = nullptr;

    // OpenGL objects
    GLuint mParticleShaderProgram = 0;
    GLuint mGridShaderProgram = 0;
    GLuint mWireframeShaderProgram = 0;
    GLuint mParticleVAO = 0, mParticleVBO = 0;
    GLuint mGridVAO = 0, mGridVBO = 0;

    // Camera parameters (default values)
    Eigen::Vector3f mCameraPosition{0.0f, 2.0f, 3.0f};
    Eigen::Vector3f mCameraTarget{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f mCameraUp{0.0f, 1.0f, 0.0f};
    float mCameraSpeed = 0.1f;

    // Grid settings
    std::vector<float> mGridVertices;
    float mGridMin = -5.0f, mGridMax = 5.0f, mGridStep = 1.0f;

    // Internal initialization functions
    bool initGLFW();
    bool initWindow();
    bool initGLAD();
    void initOpenGLSettings();
    void initShaders();
    void initGrid();
    void initParticleBuffer();

    // Process keyboard input for camera movement.
    void processInput();

    // Utility shader functions.
    GLuint compileShader(GLenum type, const char *source);
    GLuint createShaderProgram(const char *vertexSrc, const char *fragmentSrc);

    // Callback for window resizing.
    static void framebuffer_size_callback(GLFWwindow *window, int width, int height);

    // Utility functions for camera matrices.
    Eigen::Matrix4f lookAt(const Eigen::Vector3f &eye, const Eigen::Vector3f &center, const Eigen::Vector3f &up);
    Eigen::Matrix4f perspective(float fov, float aspect, float zNear, float zFar);
};

#endif // CR_TEST_VISUALIZER_H
