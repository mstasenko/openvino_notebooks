#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <utility>

namespace otc::ai::detail {
// Logical writes are bounded before growth; allocator capacity is accounted
// separately by the owning cache. Export formats may seek to rewrite headers.
class BoundedBlobBuffer final : public std::streambuf {
 public:
  explicit BoundedBlobBuffer(std::size_t limit) : limit_(limit) {}
  std::string take() { return std::move(bytes_); }
  bool failed() const { return failed_; }
 protected:
  std::streamsize xsputn(const char* data, std::streamsize count) override {
    if (count < 0 || static_cast<std::uint64_t>(count) > limit_ - position_) {
      failed_ = true;
      throw std::length_error("SeedVR2 compiled blob exceeds its bound");
    }
    const auto end = position_ + static_cast<std::size_t>(count);
    if (end > bytes_.size()) {
      try {
        bytes_.resize(end);
      } catch (const std::bad_alloc&) {
        failed_ = true;
        throw;
      }
    }
    if (count) std::copy_n(data, static_cast<std::size_t>(count), bytes_.data() + position_);
    position_ = end;
    return count;
  }
  int_type overflow(int_type value) override {
    if (traits_type::eq_int_type(value, traits_type::eof())) return traits_type::not_eof(value);
    const char c = traits_type::to_char_type(value);
    xsputn(&c, 1);
    return value;
  }
  pos_type seekoff(off_type offset, std::ios_base::seekdir direction,
                   std::ios_base::openmode mode) override {
    if (!(mode & std::ios_base::out) || (mode & std::ios_base::in) ||
        (direction != std::ios_base::beg && direction != std::ios_base::cur &&
         direction != std::ios_base::end)) return pos_type(off_type(-1));
    const off_type base = direction == std::ios_base::beg ? 0 :
        static_cast<off_type>(direction == std::ios_base::cur ? position_ : bytes_.size());
    if (offset < -base || offset > static_cast<off_type>(bytes_.size()) - base)
      return pos_type(off_type(-1));
    position_ = static_cast<std::size_t>(base + offset);
    return pos_type(position_);
  }
  pos_type seekpos(pos_type position, std::ios_base::openmode mode) override {
    return seekoff(static_cast<off_type>(position), std::ios_base::beg, mode);
  }
 private:
  std::string bytes_;
  std::size_t limit_, position_ = 0;
  bool failed_ = false;
};

// Read-only process-local bytes: no import stream copy or caller/disk blobs.
class BlobInputBuffer final : public std::streambuf {
 public:
  BlobInputBuffer(const char* data, std::size_t size) {
    if (!data || !size || size > static_cast<std::size_t>(std::numeric_limits<off_type>::max())) {
      throw std::invalid_argument("invalid compiled blob view");
    }
    auto* begin = const_cast<char*>(data);
    setg(begin, begin, begin + size);
  }
 protected:
  pos_type seekoff(off_type offset, std::ios_base::seekdir direction,
                   std::ios_base::openmode mode) override {
    if (!(mode & std::ios_base::in) || (mode & std::ios_base::out) ||
        (direction != std::ios_base::beg && direction != std::ios_base::cur &&
         direction != std::ios_base::end)) return pos_type(off_type(-1));
    const off_type size = egptr() - eback();
    const off_type base = direction == std::ios_base::beg ? 0 :
        direction == std::ios_base::cur ? gptr() - eback() : size;
    if (offset < -base || offset > size - base) return pos_type(off_type(-1));
    setg(eback(), eback() + base + offset, egptr());
    return pos_type(base + offset);
  }
  pos_type seekpos(pos_type position, std::ios_base::openmode mode) override {
    return seekoff(static_cast<off_type>(position), std::ios_base::beg, mode);
  }
};

}  // namespace otc::ai::detail
