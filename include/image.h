#pragma once
// Minimal image I/O on top of stb. Always 8-bit RGB, interleaved.
#include <cstdint>
#include <string>
#include <vector>

struct Image {
  int w = 0, h = 0;
  std::vector<uint8_t> px;  // w * h * 3
  size_t npixels() const { return (size_t)w * h; }
  size_t bytes() const { return npixels() * 3; }
};

bool load_image(const std::string& path, Image& img);
bool save_image(const std::string& path, const Image& img);  // .png or .jpg by extension
