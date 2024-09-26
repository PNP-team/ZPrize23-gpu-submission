#pragma once
#include "strobe.h"
#include "../serialize.cuh"

#define MERLIN_PROTOCOL_LABEL "Merlin v1.0"

std::vector<uint8_t> encode_usize_as_u32(size_t x) {
    assert(x <= static_cast<size_t>(UINT32_MAX));

    std::vector<uint8_t> buf(4);

    uint32_t value = static_cast<uint32_t>(x);
    buf[0] = static_cast<uint8_t>(value & 0xff);
    buf[1] = static_cast<uint8_t>((value >> 8) & 0xff);
    buf[2] = static_cast<uint8_t>((value >> 16) & 0xff);
    buf[3] = static_cast<uint8_t>((value >> 24) & 0xff);

    return buf;
}

class Transcript {
public:
    Strobe128 strobe;
    Transcript(std::string label){
        Strobe128 strobe_ = Strobe128::new_instance(MERLIN_PROTOCOL_LABEL);
        strobe = strobe_;
        append_message("dom-sep", label);
    }
    
    void append_message(std::string label, std::string message) {
        std::vector<uint8_t> data_len = encode_usize_as_u32(message.size());
        std::vector<uint8_t> data = str_to_u8(label);
        strobe.meta_ad(data, false);
        strobe.meta_ad(data_len, true);
        std::vector<uint8_t> data2 = str_to_u8(message);
        strobe.ad(data2, false);
    }

    void append_pi(std::string label, SyncedMemory item, size_t pos) {
        std::vector<uint8_t> buf(48);
        SyncedMemory item_field = to_mont(item);
        serialize(buf, BTreeMap::new_instance(item_field, pos), EmptyFlags(0));
        append_message(label, std::string(reinterpret_cast<char*>(buf.data()), buf.size()));
    }
    
    void append(char* label, SyncedMemory item) {
        std::vector<uint8_t> buf(item.size());
        serialize(buf, item, EmptyFlags(0));
        append_message(label, std::string(reinterpret_cast<char*>(buf.data()), buf.size()));
    }

    void append(char* label, AffinePointG1 item) {
        std::vector<uint8_t> buf(48);
        serialize(buf, item, EmptyFlags(0));
        append_message(label, std::string(reinterpret_cast<char*>(buf.data()), buf.size()));
    }

    void challenge_bytes(std::string label, std::vector<uint8_t>& dest) {
        std::vector<uint8_t> data_len = encode_usize_as_u32(dest.size());
        std::vector<uint8_t> data = str_to_u8(label);
        strobe.meta_ad(data, false);
        strobe.meta_ad(data_len, true);
        strobe.prf(dest, false);
    }

    SyncedMemory challenge_scalar(std::string label) {
        size_t size = fr::MODULUS_BITS / 8;
        std::vector<uint8_t> buf(size, 0);
        challenge_bytes(label, buf);
        SyncedMemory c_s = deserialize(buf, size);
        return c_s;
    }
};

