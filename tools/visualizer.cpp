// BSD 3-Clause License
//
// Copyright (c) 2025, Yafei Ou and Mahdi Tavakoli
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "visualizer.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Utility functions for camera matrices ---
Eigen::Matrix4f Visualizer::lookAt(const Eigen::Vector3f &eye, const Eigen::Vector3f &center, const Eigen::Vector3f &up)
{
    Eigen::Vector3f f = (center - eye).normalized();
    Eigen::Vector3f u = up.normalized();
    Eigen::Vector3f s = f.cross(u).normalized();
    u = s.cross(f);

    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
    view.block<1, 3>(0, 0) = s.transpose();
    view.block<1, 3>(1, 0) = u.transpose();
    view.block<1, 3>(2, 0) = -f.transpose();

    view(0, 3) = -s.dot(eye);
    view(1, 3) = -u.dot(eye);
    view(2, 3) = f.dot(eye);

    return view;
}

Eigen::Matrix4f Visualizer::perspective(float fov, float aspect, float zNear, float zFar)
{
    Eigen::Matrix4f proj = Eigen::Matrix4f::Zero();
    float tanHalfFov = std::tan(fov / 2.0f);

    proj(0, 0) = 1.0f / (aspect * tanHalfFov);
    proj(1, 1) = 1.0f / tanHalfFov;
    proj(2, 2) = -(zFar + zNear) / (zFar - zNear);
    proj(2, 3) = -2.0f * zFar * zNear / (zFar - zNear);
    proj(3, 2) = -1.0f;

    return proj;
}

// --- Shader sources ---
const char *vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * vec4(aPos, 1.0);
    gl_PointSize = 8.0;
}
)";

const char *fragmentShaderSourceParticles = R"(
#version 330 core
out vec4 FragColor;
uniform vec4 color;
void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) {
        discard;
    }
    FragColor = color;
}
)";

const char *fragmentShaderSourceGrid = R"(
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(0.3, 0.3, 0.3, 1.0);
}
)";

const char *fragmentShaderSourceWireframe = R"(
#version 330 core
out vec4 FragColor;
uniform vec4 color;
void main() {
    FragColor = color;
}
)";

const char *vertexShaderSourceTrimeshWireframe = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    gl_PointSize = 2.0;
}
)";

// --- Constructor and Destructor ---
Visualizer::Visualizer(int width, int height, const std::string &title)
    : mWidth(width), mHeight(height), mTitle(title)
{
    if (!initGLFW() || !initWindow() || !initGLAD())
    {
        std::cerr << "Initialization error in Visualizer" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    initOpenGLSettings();
    initShaders();
    initGrid();
    initParticleBuffer();
}

Visualizer::~Visualizer()
{
    glDeleteVertexArrays(1, &mParticleVAO);
    glDeleteBuffers(1, &mParticleVBO);
    glDeleteVertexArrays(1, &mGridVAO);
    glDeleteBuffers(1, &mGridVBO);
    glDeleteProgram(mParticleShaderProgram);
    glDeleteProgram(mGridShaderProgram);
    glfwDestroyWindow(mWindow);
    glfwTerminate();
}

// --- Initialization Functions ---
bool Visualizer::initGLFW()
{
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    return true;
}

bool Visualizer::initWindow()
{
    mWindow = glfwCreateWindow(mWidth, mHeight, mTitle.c_str(), nullptr, nullptr);
    if (!mWindow)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(mWindow);
    glfwSetFramebufferSizeCallback(mWindow, framebuffer_size_callback);
    return true;
}

bool Visualizer::initGLAD()
{
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return false;
    }
    return true;
}

void Visualizer::initOpenGLSettings()
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
}

// Compile and link shaders.
GLuint Visualizer::compileShader(GLenum type, const char *source)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Error: Shader Compilation Failed\n"
                  << infoLog << std::endl;
    }
    return shader;
}

GLuint Visualizer::createShaderProgram(const char *vertexSrc, const char *fragmentSrc)
{
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSrc);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSrc);
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    int success;
    char infoLog[512];
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Error: Shader Linking Failed\n"
                  << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return program;
}

void Visualizer::initShaders()
{
    mParticleShaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSourceParticles);
    mGridShaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSourceGrid);
    mWireframeShaderProgram = createShaderProgram(vertexShaderSourceTrimeshWireframe, fragmentShaderSourceWireframe);
}

// Create grid vertex data and upload it to the GPU.
void Visualizer::initGrid()
{
    // Create grid lines along the X and Z axes.
    for (float z = mGridMin; z <= mGridMax; z += mGridStep)
    {
        // Line along X direction at constant z.
        mGridVertices.push_back(mGridMin);
        mGridVertices.push_back(0.0f);
        mGridVertices.push_back(z);
        mGridVertices.push_back(mGridMax);
        mGridVertices.push_back(0.0f);
        mGridVertices.push_back(z);
    }
    for (float x = mGridMin; x <= mGridMax; x += mGridStep)
    {
        // Line along Z direction at constant x.
        mGridVertices.push_back(x);
        mGridVertices.push_back(0.0f);
        mGridVertices.push_back(mGridMin);
        mGridVertices.push_back(x);
        mGridVertices.push_back(0.0f);
        mGridVertices.push_back(mGridMax);
    }

    glGenVertexArrays(1, &mGridVAO);
    glGenBuffers(1, &mGridVBO);
    glBindVertexArray(mGridVAO);
    glBindBuffer(GL_ARRAY_BUFFER, mGridVBO);
    glBufferData(GL_ARRAY_BUFFER, mGridVertices.size() * sizeof(float), mGridVertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
}

// Create the particle VAO/VBO. The particle data is updated from the main loop.
void Visualizer::initParticleBuffer()
{
    glGenVertexArrays(1, &mParticleVAO);
    glGenBuffers(1, &mParticleVBO);
    glBindVertexArray(mParticleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, mParticleVBO);
    // Initially, allocate zero bytes. Data will be updated via updateParticleData().
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
}

// --- Public API Functions ---
bool Visualizer::running() const
{
    return !glfwWindowShouldClose(mWindow);
}

// Bind new particle data to the buffer.
void Visualizer::updateParticleData(const std::vector<float> &positions)
{
    glBindBuffer(GL_ARRAY_BUFFER, mParticleVBO);
    glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(float), positions.data(), GL_DYNAMIC_DRAW);
}

// Process keyboard input for simple camera movement.
void Visualizer::processInput()
{
    if (glfwGetKey(mWindow, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(mWindow, true);

    Eigen::Vector3f forward = (mCameraTarget - mCameraPosition).normalized();
    Eigen::Vector3f right = forward.cross(mCameraUp).normalized();

    if (glfwGetKey(mWindow, GLFW_KEY_W) == GLFW_PRESS)
        mCameraPosition += mCameraSpeed * forward;
    if (glfwGetKey(mWindow, GLFW_KEY_S) == GLFW_PRESS)
        mCameraPosition -= mCameraSpeed * forward;
    if (glfwGetKey(mWindow, GLFW_KEY_A) == GLFW_PRESS)
        mCameraPosition -= mCameraSpeed * right;
    if (glfwGetKey(mWindow, GLFW_KEY_D) == GLFW_PRESS)
        mCameraPosition += mCameraSpeed * right;
    if (glfwGetKey(mWindow, GLFW_KEY_Q) == GLFW_PRESS)
        mCameraPosition += mCameraSpeed * mCameraUp;
    if (glfwGetKey(mWindow, GLFW_KEY_E) == GLFW_PRESS)
        mCameraPosition -= mCameraSpeed * mCameraUp;
}

// Render a frame: update camera matrices, draw grid and particles.
void Visualizer::beginRender()
{
    processInput();

    // Compute view and projection matrices.
    Eigen::Matrix4f view = lookAt(mCameraPosition, mCameraTarget, mCameraUp);
    Eigen::Matrix4f projection = perspective(45.0f * M_PI / 180.0f, static_cast<float>(mWidth) / mHeight, 0.1f, 100.0f);

    // Clear the screen.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Visualizer::endRender()
{
    Eigen::Matrix4f view = lookAt(mCameraPosition, mCameraTarget, mCameraUp);
    Eigen::Matrix4f projection = perspective(45.0f * M_PI / 180.0f, static_cast<float>(mWidth) / mHeight, 0.1f, 100.0f);

    // Render the grid.
    glUseProgram(mGridShaderProgram);
    GLint gridViewLoc = glGetUniformLocation(mGridShaderProgram, "view");
    GLint gridProjLoc = glGetUniformLocation(mGridShaderProgram, "projection");
    glUniformMatrix4fv(gridViewLoc, 1, GL_FALSE, view.data());
    glUniformMatrix4fv(gridProjLoc, 1, GL_FALSE, projection.data());
    glBindVertexArray(mGridVAO);
    // Each line has 2 vertices. We have two sets of lines.
    int linesCount = (int)((mGridMax - mGridMin) / mGridStep + 1);
    glDrawArrays(GL_LINES, 0, linesCount * 4);

    glfwSwapBuffers(mWindow);
    glfwPollEvents();
}

void Visualizer::drawMeshWireframe(const std::vector<float> &vertices,
                                   const std::vector<unsigned int> &indices,
                                   const Eigen::Vector3f &position,
                                   const Eigen::Matrix3f &rotation,
                                   const Eigen::Vector3f &scale,
                                   const Eigen::Vector4f &color)
{
    // Create temporary OpenGL objects for the mesh.
    GLuint meshVAO, meshVBO, meshEBO;
    glGenVertexArrays(1, &meshVAO);
    glGenBuffers(1, &meshVBO);
    glGenBuffers(1, &meshEBO);

    // Bind the VAO.
    glBindVertexArray(meshVAO);

    // Upload vertex data.
    glBindBuffer(GL_ARRAY_BUFFER, meshVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    // Upload index data.
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // Set up vertex attributes.
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    // Shader program
    glUseProgram(mWireframeShaderProgram);

    // Apply scaling and rotation:
    // Combine rotation and scaling by multiplying the rotation matrix by a diagonal scale matrix.
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    model.block<3, 3>(0, 0) = rotation * scale.asDiagonal();
    model.block<3, 1>(0, 3) = position;

    // Compute view and projection matrices (using the same camera as in render()).
    Eigen::Matrix4f view = lookAt(mCameraPosition, mCameraTarget, mCameraUp);
    Eigen::Matrix4f projection = perspective(45.0f * M_PI / 180.0f, static_cast<float>(mWidth) / mHeight, 0.1f, 100.0f);

    // Set uniform values.
    GLint modelLoc = glGetUniformLocation(mWireframeShaderProgram, "model");
    GLint viewLoc = glGetUniformLocation(mWireframeShaderProgram, "view");
    GLint projLoc = glGetUniformLocation(mWireframeShaderProgram, "projection");
    GLint colorLoc = glGetUniformLocation(mWireframeShaderProgram, "color");
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, model.data());
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view.data());
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, projection.data());
    glUniform4fv(colorLoc, 1, color.data());

    // Set polygon mode to wireframe.
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // Draw the mesh using the indices.
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0);

    // Reset polygon mode to fill for subsequent drawing.
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // Unbind VAO (optional cleanup).
    glBindVertexArray(0);

    // Delete the temporary buffers.
    glDeleteBuffers(1, &meshVBO);
    glDeleteBuffers(1, &meshEBO);
    glDeleteVertexArrays(1, &meshVAO);
}

void Visualizer::drawParticles(const std::vector<float> &positions,
                               const Eigen::Vector4f &color)
{
    glBindBuffer(GL_ARRAY_BUFFER, mParticleVBO);
    glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(float), positions.data(), GL_DYNAMIC_DRAW);

    // Render particles.

    // Compute view and projection matrices.
    Eigen::Matrix4f view = lookAt(mCameraPosition, mCameraTarget, mCameraUp);
    Eigen::Matrix4f projection = perspective(45.0f * M_PI / 180.0f, static_cast<float>(mWidth) / mHeight, 0.1f, 100.0f);

    glUseProgram(mParticleShaderProgram);
    GLint viewLoc = glGetUniformLocation(mParticleShaderProgram, "view");
    GLint projLoc = glGetUniformLocation(mParticleShaderProgram, "projection");
    GLint colorLoc = glGetUniformLocation(mParticleShaderProgram, "color");
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view.data());
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, projection.data());
    glUniform4fv(colorLoc, 1, color.data());
    glBindVertexArray(mParticleVAO);
    GLint particleCount = 0;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &particleCount);
    particleCount /= (3 * sizeof(float));
    glDrawArrays(GL_POINTS, 0, particleCount);
}

// --- Callback Functions ---
void Visualizer::framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    glViewport(0, 0, width, height);
}
