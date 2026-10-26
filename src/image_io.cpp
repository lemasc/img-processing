#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "image.h"
#include "stb_image.h"
#include "stb_image_write.h"

bool load_image(const std::string& path, Image& img) {
  int n;
  uint8_t* d = stbi_load(path.c_str(), &img.w, &img.h, &n, 3);
  if (!d) return false;
  img.px.assign(d, d + img.bytes());
  stbi_image_free(d);
  return true;
}

bool save_image(const std::string& path, const Image& img) {
  bool jpg = path.size() > 4 && (path.substr(path.size() - 4) == ".jpg" || path.substr(path.size() - 5) == ".jpeg");
  if (jpg) return stbi_write_jpg(path.c_str(), img.w, img.h, 3, img.px.data(), 95) != 0;
  return stbi_write_png(path.c_str(), img.w, img.h, 3, img.px.data(), img.w * 3) != 0;
}
