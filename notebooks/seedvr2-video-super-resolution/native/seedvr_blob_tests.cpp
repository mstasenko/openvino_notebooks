#include "seedvr_component_cache.hpp"
#include "seedvr_blob.hpp"
#include <cassert>
#include <iostream>
#include <cstdlib>
#include <new>
#include <sstream>

// Fixture-only allocator failure; the production cache has no injection API.
bool failNextAllocation = false;
void* operator new(std::size_t size) {
  if (failNextAllocation) {
    failNextAllocation = false;
    throw std::bad_alloc();
  }
  if (void* memory = std::malloc(size ? size : 1)) return memory;
  throw std::bad_alloc();
}
void operator delete(void* memory) noexcept {
  std::free(memory);
}
void operator delete(void* memory, std::size_t) noexcept {
  std::free(memory);
}

using namespace otc::ai;
using namespace otc::ai::detail;

int main() {
  const std::uint64_t cgroup = kRetainedLimit + kTransientReserve + kMargin;
  const std::uint64_t global = cgroup + kGlobalSpare;
  ComponentCache cache;
  assert(!cache.admit(true, {global, std::nullopt}));
  assert(!cache.admit(true, {std::nullopt, cgroup}));
  assert(!cache.admit(false, {global, cgroup}));
  assert(!cache.admit(true, {global - 1, cgroup}));
  assert(!cache.admit(true, {global, cgroup - 1}));
  assert(cache.admit(true, {global, cgroup}));
  ComponentCache postDit;
  constexpr std::uint64_t snapshot = 6672856292ULL;
  SeedVrObservedHostMemory beforeDit{global + snapshot - 1, cgroup + snapshot - 1};
  assert(postDit.admit(true, beforeDit));
  SeedVrObservedHostMemory afterDit{*beforeDit.available - snapshot, *beforeDit.cgroupHeadroom - snapshot};
  assert(!postDit.admit(true, afterDit));
  ComponentCache overflow(UINT64_MAX);
  assert(!overflow.admit(true, {UINT64_MAX, UINT64_MAX}));
  const ComponentKey encoder{ComponentStage::Encoder, 64, 64};
  const ComponentKey edge{ComponentStage::Encoder, 40, 64};
  const ComponentKey third{ComponentStage::Encoder, 64, 40};
  const SeedVrObservedHostMemory ample{global, cgroup};
  assert(cache.store(encoder, "abcd", ample));
  assert(!cache.store(encoder, "replacement", ample));
  assert(cache.store(edge, "XYZ", ample));
  assert(cache.retainedCapacity() == 7);
  assert(!cache.wants(third));
  assert(!cache.find(third));
  assert(!cache.wants({ComponentStage::Decoder, 0, 64}));
  assert(!cache.wants({ComponentStage::Decoder, 65, 64}));
  bool unknownRejected = false;
  try {
    cache.wants({static_cast<ComponentStage>(99), 1, 1});
  } catch (const std::invalid_argument&) {
    unknownRejected = true;
  }
  assert(unknownRejected);
  assert(cache.store({ComponentStage::Decoder, 64, 64}, "d", ample));
  assert(cache.store({ComponentStage::Decoder, 40, 64}, "e", ample));
  assert(cache.store({ComponentStage::Prelude, 96, 120}, "p", ample));
  assert(cache.store({ComponentStage::Postlude, 96, 120}, "q", ample));
  assert(!cache.wants({ComponentStage::Prelude, 48, 60}));

  bool rejected = false;
  {
    const auto* cached = cache.find(encoder);
    BlobInputBuffer input(cached->data(), cached->size());
    std::istream stream(&input);
    char bytes[4];
    assert(stream.read(bytes, 4));
    assert(std::string(bytes, 4) == "abcd");
    stream.seekg(-2, std::ios::end);
    assert(stream.get() == 'c');
    stream.seekg(0, std::ios::beg);
    assert(stream.get() == 'a');
    stream.seekg(99, std::ios::beg);
    assert(stream.fail());
    stream.clear();
    stream.seekg(-1, std::ios::beg);
    assert(stream.fail());
    assert(input.pubseekoff(0, std::ios::beg, std::ios::out) == std::streampos(-1));
    assert(input.pubseekoff(0, static_cast<std::ios::seekdir>(99), std::ios::in) == std::streampos(-1));
    try {
      BlobInputBuffer empty(nullptr, 0);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
  // Release stream/vector borrowers before pressure fallback drops entries.
  assert(cache.admit(true, {global - 11, cgroup - 11}));
  assert(!cache.admit(true, {global - 12, cgroup - 11}));
  assert(cache.retainedCapacity() == 0 && !cache.find(encoder));

  ComponentCache tiny(3);
  assert(tiny.admit(true, ample));
  assert(!tiny.store(encoder, "abcd", ample));
  assert(!tiny.store(encoder, "a", {kGlobalSpare, kMargin}));
  assert(tiny.store(encoder, "abc", ample));
  assert(tiny.retainedCapacity() == 3);
  assert(!tiny.store(edge, "x", ample));
  ComponentCache allocationFallback(8);
  assert(allocationFallback.admit(true, ample));
  assert(allocationFallback.store(encoder, "a", ample));
  failNextAllocation = true;
  assert(!allocationFallback.store(edge, "x", ample));
  assert(!failNextAllocation && !allocationFallback.enabled() && allocationFallback.retainedCapacity() == 0);

  BoundedBlobBuffer output(4);
  assert(output.pubseekoff(0, std::ios::beg, std::ios::in) == std::streampos(-1));
  assert(output.pubseekoff(0, static_cast<std::ios::seekdir>(99), std::ios::out) == std::streampos(-1));
  std::ostream exported(&output);
  exported.exceptions(std::ios::badbit | std::ios::failbit);
  exported.write("abcd", 4);
  exported.seekp(1, std::ios::beg);
  exported.put('Z');
  exported.seekp(0, std::ios::end);
  rejected = false;
  try {
    exported.put('x');
  } catch (const std::exception&) {
    rejected = true;
  }
  assert(rejected && output.failed() && output.take() == "aZcd");
  BoundedBlobBuffer allocationOutput(64);
  std::ostream allocationStream(&allocationOutput);
  allocationStream.exceptions(std::ios::badbit | std::ios::failbit);
  failNextAllocation = true;
  rejected = false;
  try {
    allocationStream.write("abcdefghijklmnopqrstuvwxyz", 26);
  } catch (const std::exception&) {
    rejected = true;
  }
  assert(rejected && !failNextAllocation && allocationOutput.failed());
  std::cout << "cache fixtures PASS: thresholds/unknown/CPU/keys/capacity/pressure/seek/export bounds; no Core/GPU\n";
}
