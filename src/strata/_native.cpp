// strata._native: Zero-copy mmap .kmb v2 tape, 256-lane bit-sliced simulator,
// checkpointed causal-cone differential replay, OpenMP Z2/Z4/Z8 linear solver,
// and GF(2^64) Lovasz linear matroid-parity oracle.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using u128 = unsigned __int128;

static constexpr u8 KMB_MAGIC[16] = {
    0xd7, 0x50, 0xc7, 0xd5, 0xc3, 0x29, 0xd3, 0x26,
    0xe3, 0xcc, 0x9f, 0x68, 0x34, 0xf2, 0xb8, 0xbf
};

enum : u8 {
    T_NEG = 0x00,
    T_POP = 0x01,
    T_TAG = 0x06,
    T_Z = 0x10,
    T_X = 0x11,
    T_R = 0x12,
    T_ZPOW = 0x13,
    T_CZ = 0x20,
    T_CX = 0x21,
    T_SWAP = 0x22,
    T_CCZ = 0x30,
    T_CCX = 0x31,
    T_CSWAP = 0x32,
    T_BST0 = 0x40,
    T_BST1 = 0x41,
    T_BINV = 0x42,
    T_HMR = 0x50,
    T_PUSH = 0x81
};

static inline u8 base_op(u8 ty) { return ty & 0x7f; }
static inline int nqarg(u8 ty) { return (ty >> 4) & 3; }

static inline const char* op_mnemonic(u8 ty) {
    switch (base_op(ty)) {
        case T_NEG: return "NEG";
        case T_POP: return (ty & 0x80) ? "PUSH_CONDITION" : "POP_CONDITION";
        case T_TAG: return "INFO_TAG";
        case T_Z: return "Z";
        case T_X: return "X";
        case T_R: return "R";
        case T_ZPOW: return "Z_POW";
        case T_HMR: return "HMR";
        case T_CZ: return "CZ";
        case T_CX: return "CX";
        case T_SWAP: return "SWAP";
        case T_CCZ: return "CCZ";
        case T_CCX: return "CCX";
        case T_CSWAP: return "CSWAP";
        case T_BST0: return "BIT_STORE0";
        case T_BST1: return "BIT_STORE1";
        case T_BINV: return "BIT_INVERT";
        default: return "UNKNOWN";
    }
}

static inline int angle_level(u128 a) {
    u64 lo = static_cast<u64>(a);
    u64 hi = static_cast<u64>(a >> 64);
    if (lo) return 128 - __builtin_ctzll(lo);
    if (hi) return 64 - __builtin_ctzll(hi);
    return 0;
}

static inline u8 angle_class(u128 a) {
    int l = angle_level(a);
    return l <= 2 ? 0 : (l == 3 ? 1 : 2);
}

//Minimal standalone SHA-256 implementation for zero external dependencies.
struct Sha256 {
    u32 state[8];
    u64 bitlen;
    u8 buf[64];
    size_t buflen;

    Sha256() { reset(); }

    void reset() {
        state[0] = 0x6a09e667u; state[1] = 0xbb67ae85u;
        state[2] = 0x3c6ef372u; state[3] = 0xa54ff53au;
        state[4] = 0x510e527fu; state[5] = 0x9b05688cu;
        state[6] = 0x1f83d9abu; state[7] = 0x5be0cd19u;
        bitlen = 0;
        buflen = 0;
    }

    static inline u32 rotr(u32 x, u32 n) { return (x >> n) | (x << (32 - n)); }

    void transform(const u8 data[64]) {
        static constexpr u32 K[64] = {
            0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
            0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
            0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
            0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
            0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
            0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
            0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
            0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
        };
        u32 m[64];
        for (int i = 0; i < 16; ++i) {
            m[i] = (static_cast<u32>(data[i * 4]) << 24) |
                   (static_cast<u32>(data[i * 4 + 1]) << 16) |
                   (static_cast<u32>(data[i * 4 + 2]) << 8) |
                   static_cast<u32>(data[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            u32 s0 = rotr(m[i - 15], 7) ^ rotr(m[i - 15], 18) ^ (m[i - 15] >> 3);
            u32 s1 = rotr(m[i - 2], 17) ^ rotr(m[i - 2], 19) ^ (m[i - 2] >> 10);
            m[i] = m[i - 16] + s0 + m[i - 7] + s1;
        }
        u32 a = state[0], b = state[1], c = state[2], d = state[3];
        u32 e = state[4], f = state[5], g = state[6], h = state[7];
        for (int i = 0; i < 64; ++i) {
            u32 S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            u32 ch = (e & f) ^ ((~e) & g);
            u32 temp1 = h + S1 + ch + K[i] + m[i];
            u32 S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            u32 maj = (a & b) ^ (a & c) ^ (b & c);
            u32 temp2 = S0 + maj;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }

    void update(const u8* data, size_t len) {
        size_t i = 0;
        if (buflen > 0 && buflen + len >= 64) {
            size_t take = 64 - buflen;
            std::memcpy(buf + buflen, data, take);
            transform(buf);
            bitlen += 512;
            buflen = 0;
            i += take;
        }
        while (i + 64 <= len) {
            transform(data + i);
            bitlen += 512;
            i += 64;
        }
        if (i < len) {
            std::memcpy(buf + buflen, data + i, len - i);
            buflen += (len - i);
        }
    }

    std::string hexdigest() {
        u64 total_bits = bitlen + buflen * 8;
        buf[buflen++] = 0x80;
        if (buflen > 56) {
            std::memset(buf + buflen, 0, 64 - buflen);
            transform(buf);
            buflen = 0;
        }
        std::memset(buf + buflen, 0, 56 - buflen);
        for (int i = 0; i < 8; ++i) {
            buf[63 - i] = static_cast<u8>((total_bits >> (i * 8)) & 0xffu);
        }
        transform(buf);
        char hex[65];
        static constexpr char digits[] = "0123456789abcdef";
        for (int i = 0; i < 8; ++i) {
            u32 v = state[i];
            for (int j = 0; j < 8; ++j) {
                hex[i * 8 + j] = digits[(v >> (28 - 4 * j)) & 0xf];
            }
        }
        hex[64] = '\0';
        return std::string(hex, 64);
    }
};

static u128 parse_angle_half_turns(const std::string& s_in) {
    std::string s = s_in;
    bool neg = false;
    if (!s.empty() && (s[0] == '-' || s[0] == '+')) {
        neg = (s[0] == '-');
        s = s.substr(1);
    }
    size_t dot = s.find('.');
    if (dot == std::string::npos) {
        s += ".0";
        dot = s.find('.');
    }
    if (dot == 0 || dot + 1 >= s.size()) {
        throw std::runtime_error("Invalid Z_POW angle literal: " + s_in);
    }
    std::string ip = s.substr(0, dot);
    std::string fp = s.substr(dot + 1);
    bool odd = (ip.back() - '0') & 1;
    while (!fp.empty() && fp.back() == '0') fp.pop_back();
    u64 w0 = 0, w1 = 0;
    if (fp.empty()) {
        w1 = static_cast<u64>(odd) << 63;
    } else {
        bool sticky = fp.size() > 128;
        size_t np = sticky ? 128 : fp.size();
        u64 acc[8] = {0};
        auto step = [&](u64 d) {
            u64 rem = d;
            for (int i = 7; i >= 0; --i) {
                u128 cur = (static_cast<u128>(rem) << 64) | acc[i];
                acc[i] = static_cast<u64>(cur / 10);
                rem = static_cast<u64>(cur % 10);
            }
        };
        if (sticky) step(5);
        for (size_t k = np; k--;) step(static_cast<u64>(fp[k] - '0'));
        w1 = (acc[7] >> 1) | (static_cast<u64>(odd) << 63);
        w0 = (acc[7] << 63) | (acc[6] >> 1);
        if (acc[6] & 1) {
            bool sl = acc[5] || acc[4] || acc[3] || acc[2] || acc[1] || acc[0];
            if (sl || (w0 & 1)) {
                w0 += 1;
                if (w0 == 0) w1 += 1;
            }
        }
    }
    u128 v = (static_cast<u128>(w1) << 64) | w0;
    return neg ? (static_cast<u128>(0) - v) : v;
}

// Convert u128 turn units back to canonical decimal half-turns (2 * turns mod 2).
static std::string format_angle_half_turns(u128 a) {
    // Half-turns = 2 * (a / 2^128) in [0, 2).
    // Integer part is top bit of a (0 or 1); fractional part is (a & ((1<<127)-1)) / 2^127.
    u32 int_part = static_cast<u32>((a >> 127) & 1u);
    u128 frac = (a << 1);  // fractional part in units of 2^-128
    if (frac == 0) {
        return std::to_string(int_part) + ".0";
    }
    // Extract decimal digits via multiplication by 10; find shortest exact / 40-digit representation.
    std::string digits;
    digits.reserve(44);
    u128 cur = frac;
    for (int d = 0; d < 42 && cur != 0; ++d) {
        u64 lo = static_cast<u64>(cur);
        u64 hi = static_cast<u64>(cur >> 64);
        u128 lo10 = static_cast<u128>(lo) * 10u;
        u128 hi10 = static_cast<u128>(hi) * 10u + (lo10 >> 64);
        u32 dig = static_cast<u32>(hi10 >> 64);
        digits.push_back(static_cast<char>('0' + dig));
        cur = (static_cast<u128>(static_cast<u64>(hi10)) << 64) | static_cast<u64>(lo10);
    }
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    return std::to_string(int_part) + "." + digits;
}

struct BlockIdx {
    u64 c3 = 0, c2 = 0, c1 = 0, cbt = 0, cbc = 0, cang = 0, ctag = 0;
    int32_t depth = 0;
};

struct OpRec {
    u8 ty = 0, fl = 0;
    u16 q0 = 0, q1 = 0, q2 = 0;
    u32 bt = 0, bc = 0;
};

// Zero-copy mmap columnar circuit container supporting both .kmb v2 and .kmx text.
struct ColumnarCircuit {
    static constexpr size_t BLOCK_STRIDE = 1024;

    std::string path;
    std::string sha256;
    size_t file_bytes = 0;
    bool is_kmb = false;
    double open_ms = 0.0;

    // mmap backing when is_kmb == true
    int fd = -1;
    const u8* mapped = nullptr;
    size_t mapped_len = 0;

    u32 nq = 0, nb = 0;
    std::vector<std::pair<std::string, std::vector<u32>>> regs;

    // Owned storage when parsed from .kmx or memory string
    std::vector<u8> owned_types;
    std::vector<u32> owned_a0, owned_a1, owned_a2;
    std::vector<u32> owned_b0, owned_b1;
    std::vector<u32> owned_q0;
    std::vector<u32> owned_bt, owned_bc;
    std::vector<u8> owned_angles_raw;  // 16 bytes per angle

    // Columnar pointers (point into mmap for .kmb, or owned_* for .kmx)
    u64 num_ops = 0;
    u64 cnt[9] = {0};
    const u8* types = nullptr;
    const u32* a0 = nullptr;
    const u32* a1 = nullptr;
    const u32* a2 = nullptr;
    const u32* b0 = nullptr;
    const u32* b1 = nullptr;
    const u32* q0 = nullptr;
    const u32* bt = nullptr;
    const u32* bc = nullptr;
    const u8* ang_raw = nullptr;

    std::vector<u128> angles;
    std::vector<u8> angle_cls;
    std::vector<std::string> tags;
    std::vector<BlockIdx> blocks;

    u64 static_toffoli = 0;
    u64 static_t = 0;
    u64 static_rot = 0;
    std::array<u64, 256> opcode_hist{};

    ~ColumnarCircuit() {
        if (mapped && mapped_len > 0) {
            munmap(const_cast<u8*>(mapped), mapped_len);
            mapped = nullptr;
        }
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
    }

    void build_block_index() {
        size_t nblocks = static_cast<size_t>(num_ops / BLOCK_STRIDE) + 1;
        blocks.resize(nblocks);
        opcode_hist.fill(0);
        BlockIdx cur{};
        for (u64 k = 0; k < num_ops; ++k) {
            if ((k & (BLOCK_STRIDE - 1)) == 0) {
                blocks[static_cast<size_t>(k / BLOCK_STRIDE)] = cur;
            }
            u8 ty = types[k];
            opcode_hist[ty]++;
            int nqa = nqarg(ty);
            if (nqa == 3) cur.c3++;
            else if (nqa == 2) cur.c2++;
            else if (nqa == 1) cur.c1++;
            if (ty & 0x40) cur.cbt++;
            if (ty & 0x80) cur.cbc++;
            if (base_op(ty) == T_ZPOW) cur.cang++;
            if (ty == T_TAG) cur.ctag++;
            if (ty == T_PUSH) cur.depth++;
            else if (ty == T_POP && cur.depth > 0) cur.depth--;
        }
        static_toffoli = 0;
        for (int ty = 0; ty < 256; ++ty) {
            u8 b = base_op(static_cast<u8>(ty));
            if (b == T_CCX || b == T_CCZ || b == T_CSWAP) {
                static_toffoli += opcode_hist[ty];
            }
        }
        static_t = 0;
        static_rot = 0;
        for (u8 cl : angle_cls) {
            if (cl == 1) static_t++;
            else if (cl == 2) static_rot++;
        }
    }

    BlockIdx seek_cursor(u64 target_op) const {
        if (target_op > num_ops) throw std::out_of_range("op index out of range");
        size_t blk = static_cast<size_t>(target_op / BLOCK_STRIDE);
        BlockIdx cur = blocks[blk];
        u64 k0 = static_cast<u64>(blk) * BLOCK_STRIDE;
        for (u64 k = k0; k < target_op; ++k) {
            u8 ty = types[k];
            int nqa = nqarg(ty);
            if (nqa == 3) cur.c3++;
            else if (nqa == 2) cur.c2++;
            else if (nqa == 1) cur.c1++;
            if (ty & 0x40) cur.cbt++;
            if (ty & 0x80) cur.cbc++;
            if (base_op(ty) == T_ZPOW) cur.cang++;
            if (ty == T_TAG) cur.ctag++;
            if (ty == T_PUSH) cur.depth++;
            else if (ty == T_POP && cur.depth > 0) cur.depth--;
        }
        return cur;
    }

    void extract_ops(u64 start, u64 stop, std::vector<OpRec>& out) const {
        if (start > stop || stop > num_ops) throw std::out_of_range("invalid span");
        out.resize(static_cast<size_t>(stop - start));
        BlockIdx cur = seek_cursor(start);
        for (u64 k = start; k < stop; ++k) {
            u8 ty = types[k];
            OpRec o{};
            o.ty = ty;
            int nqa = nqarg(ty);
            if (nqa == 3) {
                o.q0 = static_cast<u16>(a0[cur.c3]);
                o.q1 = static_cast<u16>(a1[cur.c3]);
                o.q2 = static_cast<u16>(a2[cur.c3]);
                cur.c3++;
            } else if (nqa == 2) {
                o.q0 = static_cast<u16>(b0[cur.c2]);
                o.q1 = static_cast<u16>(b1[cur.c2]);
                cur.c2++;
            } else if (nqa == 1) {
                o.q0 = static_cast<u16>(q0[cur.c1]);
                cur.c1++;
            }
            if (ty & 0x40) o.bt = bt[cur.cbt++];
            if (ty & 0x80) o.bc = bc[cur.cbc++];
            if (base_op(ty) == T_ZPOW) o.bt = static_cast<u32>(cur.cang++);
            if (ty == T_TAG) o.bt = static_cast<u32>(cur.ctag++);
            out[static_cast<size_t>(k - start)] = o;
        }
    }

    std::string format_op(const OpRec& o) const {
        u8 ty = o.ty;
        u8 b = base_op(ty);
        if (ty == T_TAG) {
            std::string escaped;
            const std::string& t = tags.at(o.bt);
            escaped.push_back('"');
            for (char ch : t) {
                if (ch == '"' || ch == '\\') escaped.push_back('\\');
                escaped.push_back(ch);
            }
            escaped.push_back('"');
            return "INFO_TAG " + escaped;
        }
        if (ty == T_PUSH) {
            return "PUSH_CONDITION if b" + std::to_string(o.bc);
        }
        if (ty == T_POP) {
            return "POP_CONDITION";
        }
        std::string s = op_mnemonic(ty);
        int nqa = nqarg(ty);
        if (nqa >= 1) s += " q" + std::to_string(o.q0);
        if (nqa >= 2) s += " q" + std::to_string(o.q1);
        if (nqa >= 3) s += " q" + std::to_string(o.q2);
        if (ty & 0x40) s += " b" + std::to_string(o.bt);
        if (b == T_ZPOW) s += " " + format_angle_half_turns(angles.at(o.bt));
        if (ty & 0x80) s += " if b" + std::to_string(o.bc);
        return s;
    }

    void parse_kmb_buffer(const u8* d, size_t n) {
        is_kmb = true;
        u32 ver = 0, nreg = 0;
        std::memcpy(&ver, d + 16, 4);
        std::memcpy(&nq, d + 20, 4);
        std::memcpy(&nb, d + 24, 4);
        std::memcpy(&nreg, d + 28, 4);
        std::memcpy(cnt, d + 32, 72);
        if (ver != 2) throw std::runtime_error("Unsupported .kmb version");
        if (nq > 65535) throw std::runtime_error("More than 65535 qubits not supported");

        size_t pos = 104;
        const u32* pk[10] = {nullptr};
        u64 pl[12] = {0};
        const u8* tagp = nullptr;
        while (pos < n) {
            if (pos + 16 > n) throw std::runtime_error("Truncated .kmb packet header");
            u64 pt = 0, len = 0;
            std::memcpy(&pt, d + pos, 8);
            std::memcpy(&len, d + pos + 8, 8);
            pos += 16;
            if (pos + len > n) throw std::runtime_error("Truncated .kmb packet payload");
            if (pt == 0) {
                if (len < 8) throw std::runtime_error("Malformed register packet");
                u32 nl = 0, rl = 0;
                std::memcpy(&nl, d + pos, 4);
                std::memcpy(&rl, d + pos + 4, 4);
                if (8ull + nl + 4ull * rl > len) throw std::runtime_error("Register packet overflow");
                std::vector<u32> ids(rl);
                if (rl) std::memcpy(ids.data(), d + pos + 8 + nl, 4ull * rl);
                regs.emplace_back(std::string(reinterpret_cast<const char*>(d + pos + 8), nl), std::move(ids));
            } else if (pt == 1) {
                types = d + pos;
                pl[1] = len;
            } else if (pt >= 2 && pt <= 9) {
                pk[pt] = reinterpret_cast<const u32*>(d + pos);
                pl[pt] = len;
            } else if (pt == 10) {
                ang_raw = d + pos;
                pl[10] = len;
            } else if (pt == 11) {
                tagp = d + pos;
                pl[11] = len;
            }
            pos += len;
        }
        if (cnt[0] > 0 && (!types || pl[1] != cnt[0])) {
            throw std::runtime_error("Invalid .kmb op type packet length");
        }
        static const u8 empty_u8 = 0;
        static const u32 empty_u32 = 0;
        num_ops = cnt[0];
        if (!types) types = &empty_u8;
        a0 = pk[2] ? pk[2] : &empty_u32;
        a1 = pk[3] ? pk[3] : &empty_u32;
        a2 = pk[4] ? pk[4] : &empty_u32;
        b0 = pk[5] ? pk[5] : &empty_u32;
        b1 = pk[6] ? pk[6] : &empty_u32;
        q0 = pk[7] ? pk[7] : &empty_u32;
        bt = pk[8] ? pk[8] : &empty_u32;
        bc = pk[9] ? pk[9] : &empty_u32;
        if (!ang_raw) ang_raw = &empty_u8;

        size_t na = static_cast<size_t>(pl[10] / 16);
        angles.resize(na);
        angle_cls.resize(na);
        for (size_t i = 0; i < na; ++i) {
            u64 w[2];
            std::memcpy(w, ang_raw + 16 * i, 16);
            angles[i] = (static_cast<u128>(w[1]) << 64) | w[0];
            angle_cls[i] = angle_class(angles[i]);
        }
        if (tagp && pl[11] >= 8) {
            u64 nt = 0;
            std::memcpy(&nt, tagp, 8);
            const u64* eo = reinterpret_cast<const u64*>(tagp + 8);
            const char* ch = reinterpret_cast<const char*>(tagp + 8 * (nt + 1));
            u64 p = 0;
            for (u64 i = 0; i < nt; ++i) {
                u64 e = 0;
                std::memcpy(&e, eo + i, 8);
                tags.emplace_back(ch + p, static_cast<size_t>(e - p));
                p = e;
            }
        }
        build_block_index();
    }

    void parse_kmx_buffer(const char* d, size_t n) {
        is_kmb = false;
        std::map<u32, std::vector<u32>> rmem;
        std::map<u32, std::string> rname;
        u32 mq = 0, mb = 0;
        bool anyq = false, anyb = false;

        static const std::map<std::string, u8> NM = {
            {"X", T_X}, {"CX", T_CX}, {"CCX", T_CCX},
            {"Z", T_Z}, {"CZ", T_CZ}, {"CCZ", T_CCZ},
            {"Z_POW", T_ZPOW}, {"NEG", T_NEG},
            {"SWAP", T_SWAP}, {"CSWAP", T_CSWAP},
            {"R", T_R}, {"HMR", T_HMR},
            {"BIT_INVERT", T_BINV}, {"BIT_STORE0", T_BST0}, {"BIT_STORE1", T_BST1}
        };

        size_t p = 0;
        while (p < n) {
            size_t e = p;
            while (e < n && d[e] != '\n') ++e;
            std::string L(d + p, e - p);
            if (!L.empty() && L.back() == '\r') L.pop_back();
            p = e + 1;

            // Extract quoted string before '#' if any
            std::string quoted;
            size_t qa = L.find('"');
            size_t h = L.find('#');
            if (h != std::string::npos && (qa == std::string::npos || h < qa)) {
                L.resize(h);
            } else if (qa != std::string::npos) {
                size_t qb = qa + 1;
                while (qb < L.size()) {
                    if (L[qb] == '\\' && qb + 1 < L.size()) {
                        quoted.push_back(L[qb + 1]);
                        qb += 2;
                    } else if (L[qb] == '"') {
                        break;
                    } else {
                        quoted.push_back(L[qb++]);
                    }
                }
                L.resize(qa);
            }

            std::vector<std::string> tk;
            {
                std::istringstream is(L);
                std::string t;
                while (is >> t) tk.push_back(std::move(t));
            }
            if (tk.empty()) continue;
            const std::string& nm = tk[0];
            if (nm == "DEBUG_PRINT") continue;

            std::vector<u32> qs, bs, rs;
            std::string ang;
            bool cond = false;
            u32 cb = 0;
            for (size_t i = 1; i < tk.size(); ++i) {
                const std::string& t = tk[i];
                if (t == "if") {
                    cond = true;
                    if (i + 1 >= tk.size() || tk[i + 1][0] != 'b') {
                        throw std::runtime_error("Malformed condition in .kmx");
                    }
                    cb = static_cast<u32>(std::strtoul(tk[i + 1].c_str() + 1, nullptr, 10));
                    ++i;
                    continue;
                }
                if (t[0] == 'q' && t.size() > 1 && std::isdigit(static_cast<unsigned char>(t[1]))) {
                    qs.push_back(static_cast<u32>(std::strtoul(t.c_str() + 1, nullptr, 10)));
                } else if (t[0] == 'b' && t.size() > 1 && std::isdigit(static_cast<unsigned char>(t[1]))) {
                    bs.push_back(static_cast<u32>(std::strtoul(t.c_str() + 1, nullptr, 10)));
                } else if (t[0] == 'r' && t.size() > 1 && std::isdigit(static_cast<unsigned char>(t[1]))) {
                    rs.push_back(static_cast<u32>(std::strtoul(t.c_str() + 1, nullptr, 10)));
                } else {
                    ang = t;
                }
            }
            for (u32 q : qs) { mq = std::max(mq, q); anyq = true; }
            for (u32 b : bs) { mb = std::max(mb, b); anyb = true; }
            if (cond) { mb = std::max(mb, cb); anyb = true; }

            if (nm == "APPEND_TO_REGISTER") {
                if (rs.size() != 1) throw std::runtime_error("Invalid APPEND_TO_REGISTER");
                if (!qs.empty()) rmem[rs[0]].push_back(qs[0] | 0x80000000u);
                else if (!bs.empty()) rmem[rs[0]].push_back(bs[0]);
                continue;
            }
            if (nm == "REGISTER") {
                if (rs.size() != 1) throw std::runtime_error("Invalid REGISTER");
                rmem[rs[0]];
                rname[rs[0]] = quoted;
                continue;
            }
            if (nm == "INFO_TAG") {
                owned_types.push_back(T_TAG);
                tags.push_back(quoted);
                continue;
            }
            if (nm == "PUSH_CONDITION") {
                owned_types.push_back(T_PUSH);
                owned_bc.push_back(cb);
                continue;
            }
            if (nm == "POP_CONDITION") {
                owned_types.push_back(T_POP);
                continue;
            }
            auto it = NM.find(nm);
            if (it == NM.end()) {
                throw std::runtime_error("Unknown instruction in .kmx: " + nm);
            }
            u8 ty = static_cast<u8>(it->second | (cond ? 0x80 : 0));
            int nqa = nqarg(ty);
            if (static_cast<int>(qs.size()) != nqa) {
                throw std::runtime_error("Arity mismatch for " + nm);
            }
            owned_types.push_back(ty);
            if (nqa == 3) {
                owned_a0.push_back(qs[0]); owned_a1.push_back(qs[1]); owned_a2.push_back(qs[2]);
            } else if (nqa == 2) {
                owned_b0.push_back(qs[0]); owned_b1.push_back(qs[1]);
            } else if (nqa == 1) {
                owned_q0.push_back(qs[0]);
            }
            if (ty & 0x40) {
                if (bs.size() != 1) throw std::runtime_error("Expected 1 bit target for " + nm);
                owned_bt.push_back(bs[0]);
            }
            if (ty & 0x80) {
                owned_bc.push_back(cb);
            }
            if (it->second == T_ZPOW) {
                u128 aval = parse_angle_half_turns(ang);
                angles.push_back(aval);
                angle_cls.push_back(angle_class(aval));
                u64 w[2] = {static_cast<u64>(aval), static_cast<u64>(aval >> 64)};
                size_t old_sz = owned_angles_raw.size();
                owned_angles_raw.resize(old_sz + 16);
                std::memcpy(owned_angles_raw.data() + old_sz, w, 16);
            }
        }
        nq = anyq ? (mq + 1) : 0;
        nb = anyb ? (mb + 1) : 0;
        if (nq > 65535) throw std::runtime_error("More than 65535 qubits not supported");
        for (auto& kv : rmem) {
            regs.emplace_back(rname.count(kv.first) ? rname[kv.first] : std::string(), std::move(kv.second));
        }

        num_ops = owned_types.size();
        cnt[0] = num_ops;
        cnt[1] = owned_a0.size();
        cnt[2] = owned_b0.size();
        cnt[3] = owned_q0.size();
        cnt[4] = owned_bt.size();
        cnt[5] = owned_bc.size();
        cnt[6] = angles.size();
        cnt[7] = tags.size();
        u64 tag_bytes = 0;
        for (const auto& t : tags) tag_bytes += t.size();
        cnt[8] = tag_bytes;

        static const u8 empty_u8 = 0;
        static const u32 empty_u32 = 0;
        types = owned_types.empty() ? &empty_u8 : owned_types.data();
        a0 = owned_a0.empty() ? &empty_u32 : owned_a0.data();
        a1 = owned_a1.empty() ? &empty_u32 : owned_a1.data();
        a2 = owned_a2.empty() ? &empty_u32 : owned_a2.data();
        b0 = owned_b0.empty() ? &empty_u32 : owned_b0.data();
        b1 = owned_b1.empty() ? &empty_u32 : owned_b1.data();
        q0 = owned_q0.empty() ? &empty_u32 : owned_q0.data();
        bt = owned_bt.empty() ? &empty_u32 : owned_bt.data();
        bc = owned_bc.empty() ? &empty_u32 : owned_bc.data();
        ang_raw = owned_angles_raw.empty() ? &empty_u8 : owned_angles_raw.data();
        build_block_index();
    }

    static std::shared_ptr<ColumnarCircuit> from_file(const std::string& p) {
        auto t0 = std::chrono::steady_clock::now();
        auto c = std::make_shared<ColumnarCircuit>();
        c->path = p;
        int fd = ::open(p.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Cannot open circuit file: " + p);
        struct stat sb{};
        if (fstat(fd, &sb) != 0) {
            ::close(fd);
            throw std::runtime_error("Cannot stat circuit file: " + p);
        }
        size_t n = static_cast<size_t>(sb.st_size);
        c->file_bytes = n;
        if (n == 0) {
            ::close(fd);
            Sha256 hasher;
            c->sha256 = hasher.hexdigest();
            c->parse_kmx_buffer("", 0);
        } else {
            const u8* d = static_cast<const u8*>(::mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0));
            if (d == MAP_FAILED) {
                ::close(fd);
                throw std::runtime_error("mmap failed for: " + p);
            }
            Sha256 hasher;
            hasher.update(d, n);
            c->sha256 = hasher.hexdigest();
            if (n >= 104 && std::memcmp(d, KMB_MAGIC, 16) == 0) {
                c->fd = fd;
                c->mapped = d;
                c->mapped_len = n;
                c->parse_kmb_buffer(d, n);
            } else {
                try {
                    c->parse_kmx_buffer(reinterpret_cast<const char*>(d), n);
                } catch (...) {
                    ::munmap(const_cast<u8*>(d), n);
                    ::close(fd);
                    throw;
                }
                ::munmap(const_cast<u8*>(d), n);
                ::close(fd);
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        c->open_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return c;
    }

    static std::shared_ptr<ColumnarCircuit> from_kmx_text(const std::string& text) {
        auto t0 = std::chrono::steady_clock::now();
        auto c = std::make_shared<ColumnarCircuit>();
        c->file_bytes = text.size();
        Sha256 hasher;
        hasher.update(reinterpret_cast<const u8*>(text.data()), text.size());
        c->sha256 = hasher.hexdigest();
        c->parse_kmx_buffer(text.data(), text.size());
        auto t1 = std::chrono::steady_clock::now();
        c->open_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return c;
    }

    bool same_span(u64 s1, u64 e1, const ColumnarCircuit& other, u64 s2, u64 e2) const {
        if (s1 > e1 || e1 > num_ops || s2 > e2 || e2 > other.num_ops) {
            throw std::out_of_range("span out of range");
        }
        u64 len = e1 - s1;
        if (e2 - s2 != len) return false;
        if (len == 0) return true;
        if (std::memcmp(types + s1, other.types + s2, static_cast<size_t>(len)) != 0) return false;

        BlockIdx a_beg = seek_cursor(s1), a_end = seek_cursor(e1);
        BlockIdx b_beg = other.seek_cursor(s2);

        auto cmp_u32 = [](const u32* p, u64 o1, const u32* q, u64 o2, u64 count) {
            return count == 0 || std::memcmp(p + o1, q + o2, static_cast<size_t>(count * 4)) == 0;
        };
        if (!cmp_u32(a0, a_beg.c3, other.a0, b_beg.c3, a_end.c3 - a_beg.c3)) return false;
        if (!cmp_u32(a1, a_beg.c3, other.a1, b_beg.c3, a_end.c3 - a_beg.c3)) return false;
        if (!cmp_u32(a2, a_beg.c3, other.a2, b_beg.c3, a_end.c3 - a_beg.c3)) return false;
        if (!cmp_u32(b0, a_beg.c2, other.b0, b_beg.c2, a_end.c2 - a_beg.c2)) return false;
        if (!cmp_u32(b1, a_beg.c2, other.b1, b_beg.c2, a_end.c2 - a_beg.c2)) return false;
        if (!cmp_u32(q0, a_beg.c1, other.q0, b_beg.c1, a_end.c1 - a_beg.c1)) return false;
        if (!cmp_u32(bt, a_beg.cbt, other.bt, b_beg.cbt, a_end.cbt - a_beg.cbt)) return false;
        if (!cmp_u32(bc, a_beg.cbc, other.bc, b_beg.cbc, a_end.cbc - a_beg.cbc)) return false;

        u64 nang = a_end.cang - a_beg.cang;
        if (nang > 0 && std::memcmp(ang_raw + 16 * a_beg.cang, other.ang_raw + 16 * b_beg.cang, static_cast<size_t>(16 * nang)) != 0) {
            return false;
        }
        u64 ntag = a_end.ctag - a_beg.ctag;
        for (u64 i = 0; i < ntag; ++i) {
            if (tags[static_cast<size_t>(a_beg.ctag + i)] != other.tags[static_cast<size_t>(b_beg.ctag + i)]) {
                return false;
            }
        }
        return true;
    }

    std::string fragment_sha256(u64 start, u64 stop) const {
        if (start > stop || stop > num_ops) throw std::out_of_range("span out of range");
        Sha256 h;
        std::vector<OpRec> chunk;
        constexpr u64 STEP = 4096;
        for (u64 pos = start; pos < stop; pos += STEP) {
            u64 end = std::min(stop, pos + STEP);
            extract_ops(pos, end, chunk);
            for (const auto& op : chunk) {
                std::string line = format_op(op);
                line.push_back('\n');
                h.update(reinterpret_cast<const u8*>(line.data()), line.size());
            }
        }
        return h.hexdigest();
    }
};

struct SpanSeg {
    const ColumnarCircuit* src;
    u64 start;
    u64 stop;
    BlockIdx beg;
    BlockIdx end;
};

static std::string write_composed_kmb(
    const ColumnarCircuit& parent,
    const std::vector<SpanSeg>& segs,
    const std::string& out_path
) {
    u64 c[9] = {0};
    u32 max_nq = parent.nq;
    u32 max_nb = parent.nb;
    for (const auto& sg : segs) {
        max_nq = std::max(max_nq, sg.src->nq);
        max_nb = std::max(max_nb, sg.src->nb);
        c[0] += (sg.stop - sg.start);
        c[1] += (sg.end.c3 - sg.beg.c3);
        c[2] += (sg.end.c2 - sg.beg.c2);
        c[3] += (sg.end.c1 - sg.beg.c1);
        c[4] += (sg.end.cbt - sg.beg.cbt);
        c[5] += (sg.end.cbc - sg.beg.cbc);
        c[6] += (sg.end.cang - sg.beg.cang);
        u64 ntag = sg.end.ctag - sg.beg.ctag;
        c[7] += ntag;
        for (u64 i = 0; i < ntag; ++i) {
            c[8] += sg.src->tags[static_cast<size_t>(sg.beg.ctag + i)].size();
        }
    }

    FILE* fp = std::fopen(out_path.c_str(), "wb");
    if (!fp) throw std::runtime_error("Cannot open output path for writing: " + out_path);
    Sha256 hasher;
    auto emit_bytes = [&](const void* ptr, size_t len) {
        if (len == 0) return;
        if (std::fwrite(ptr, 1, len, fp) != len) {
            std::fclose(fp);
            throw std::runtime_error("Short write while emitting .kmb");
        }
        hasher.update(static_cast<const u8*>(ptr), len);
    };
    auto emit_pkt_hdr = [&](u64 pt, u64 plen) {
        u64 hdr[2] = {pt, plen};
        emit_bytes(hdr, 16);
    };

    emit_bytes(KMB_MAGIC, 16);
    u32 head[4] = {2u, max_nq, max_nb, static_cast<u32>(parent.regs.size())};
    emit_bytes(head, 16);
    emit_bytes(c, 72);

    // Packet 0: registers from parent
    for (const auto& r : parent.regs) {
        u32 nl = static_cast<u32>(r.first.size());
        u32 rl = static_cast<u32>(r.second.size());
        emit_pkt_hdr(0, 8ull + nl + 4ull * rl);
        emit_bytes(&nl, 4);
        emit_bytes(&rl, 4);
        emit_bytes(r.first.data(), nl);
        if (rl) emit_bytes(r.second.data(), 4ull * rl);
    }

    // Packet 1: types
    emit_pkt_hdr(1, c[0]);
    for (const auto& sg : segs) {
        emit_bytes(sg.src->types + sg.start, static_cast<size_t>(sg.stop - sg.start));
    }

    // Packets 2..9: u32 operand columns
    auto emit_u32_col = [&](u64 pt, u64 count, auto getter, auto idx_getter) {
        emit_pkt_hdr(pt, 4ull * count);
        for (const auto& sg : segs) {
            u64 b = idx_getter(sg.beg);
            u64 e = idx_getter(sg.end);
            if (e > b) {
                emit_bytes(getter(*sg.src) + b, static_cast<size_t>((e - b) * 4));
            }
        }
    };
    emit_u32_col(2, c[1], [](const ColumnarCircuit& x) { return x.a0; }, [](const BlockIdx& b) { return b.c3; });
    emit_u32_col(3, c[1], [](const ColumnarCircuit& x) { return x.a1; }, [](const BlockIdx& b) { return b.c3; });
    emit_u32_col(4, c[1], [](const ColumnarCircuit& x) { return x.a2; }, [](const BlockIdx& b) { return b.c3; });
    emit_u32_col(5, c[2], [](const ColumnarCircuit& x) { return x.b0; }, [](const BlockIdx& b) { return b.c2; });
    emit_u32_col(6, c[2], [](const ColumnarCircuit& x) { return x.b1; }, [](const BlockIdx& b) { return b.c2; });
    emit_u32_col(7, c[3], [](const ColumnarCircuit& x) { return x.q0; }, [](const BlockIdx& b) { return b.c1; });
    emit_u32_col(8, c[4], [](const ColumnarCircuit& x) { return x.bt; }, [](const BlockIdx& b) { return b.cbt; });
    emit_u32_col(9, c[5], [](const ColumnarCircuit& x) { return x.bc; }, [](const BlockIdx& b) { return b.cbc; });

    // Packet 10: angles (16 bytes each)
    emit_pkt_hdr(10, 16ull * c[6]);
    for (const auto& sg : segs) {
        u64 n_ang = sg.end.cang - sg.beg.cang;
        if (n_ang > 0) {
            emit_bytes(sg.src->ang_raw + 16 * sg.beg.cang, static_cast<size_t>(16 * n_ang));
        }
    }

    // Packet 11: tags
    emit_pkt_hdr(11, 8ull + 8ull * c[7] + c[8]);
    emit_bytes(&c[7], 8);
    u64 running = 0;
    for (const auto& sg : segs) {
        u64 nt = sg.end.ctag - sg.beg.ctag;
        for (u64 i = 0; i < nt; ++i) {
            running += sg.src->tags[static_cast<size_t>(sg.beg.ctag + i)].size();
            emit_bytes(&running, 8);
        }
    }
    for (const auto& sg : segs) {
        u64 nt = sg.end.ctag - sg.beg.ctag;
        for (u64 i = 0; i < nt; ++i) {
            const auto& t = sg.src->tags[static_cast<size_t>(sg.beg.ctag + i)];
            emit_bytes(t.data(), t.size());
        }
    }

    std::fclose(fp);
    return hasher.hexdigest();
}

// ---------------------------------------------------------------------------
// 256-lane Bit-Sliced Simulator & Snapshot State
// ---------------------------------------------------------------------------
struct alignas(64) Word256 {
    u64 w[4] = {0, 0, 0, 0};
};

struct SimState256 {
    std::vector<Word256> q;
    std::vector<Word256> b;
    Word256 sign_ph{};
    std::array<u128, 256> rot_ang{};
    std::vector<Word256> cond_stack;
    u64 rng[4] = {0, 0, 0, 0};
    u64 exec_tof = 0;
    u64 exec_t = 0;
    u64 exec_rot = 0;

    void init(u32 nq, u32 nb, u64 seed) {
        q.assign(nq, Word256{});
        b.assign(nb, Word256{});
        sign_ph = Word256{};
        rot_ang.fill(0);
        cond_stack.clear();
        exec_tof = exec_t = exec_rot = 0;
        for (int i = 0; i < 4; ++i) {
            seed += 0x9e3779b97f4a7c15ull;
            u64 z = seed;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
            rng[i] = z ^ (z >> 31);
        }
    }

    inline u64 next_rnd() {
        u64 r = std::rotl(rng[1] * 5, 7) * 9;
        u64 t = rng[1] << 17;
        rng[2] ^= rng[0];
        rng[3] ^= rng[1];
        rng[1] ^= rng[2];
        rng[0] ^= rng[3];
        rng[2] ^= t;
        rng[3] = std::rotl(rng[3], 45);
        return r;
    }

    // Total phase per lane in u128 turn units (sign_ph contributes 0.5 turns = 1<<127).
    inline u128 total_phase(int lane) const {
        u128 s = ((sign_ph.w[lane >> 6] >> (lane & 63)) & 1ull) ? (static_cast<u128>(1) << 127) : 0;
        return rot_ang[lane] + s;
    }
};

static void run_span_256(
    const ColumnarCircuit& circ,
    u64 start,
    u64 stop,
    SimState256& st,
    int outcomes_mode = 0,
    int active_lanes = 256,
    const std::vector<u8>* cone_q_filter = nullptr,
    const std::vector<u8>* cone_b_filter = nullptr,
    u64* ops_executed_out = nullptr
) {
    if (st.q.size() < circ.nq) st.q.resize(circ.nq, Word256{});
    if (st.b.size() < circ.nb) st.b.resize(circ.nb, Word256{});

    Word256 ones;
    ones.w[0] = ones.w[1] = ones.w[2] = ones.w[3] = ~0ull;
    Word256 active_mask{};
    for (int l = 0; l < active_lanes; ++l) {
        active_mask.w[l >> 6] |= (1ull << (l & 63));
    }

    auto popc_active = [&](const Word256& c) -> u64 {
        return static_cast<u64>(
            std::popcount(c.w[0] & active_mask.w[0]) +
            std::popcount(c.w[1] & active_mask.w[1]) +
            std::popcount(c.w[2] & active_mask.w[2]) +
            std::popcount(c.w[3] & active_mask.w[3])
        );
    };

    BlockIdx cur = circ.seek_cursor(start);
    Word256 bcond = st.cond_stack.empty() ? ones : st.cond_stack.back();
    u64 executed = 0;

    for (u64 k = start; k < stop; ++k) {
        u8 ty = circ.types[k];
        int nqa = nqarg(ty);
        u16 q_0 = 0, q_1 = 0, q_2 = 0;
        u32 b_t = 0, b_c = 0, idx_extra = 0;
        if (nqa == 3) {
            q_0 = static_cast<u16>(circ.a0[cur.c3]);
            q_1 = static_cast<u16>(circ.a1[cur.c3]);
            q_2 = static_cast<u16>(circ.a2[cur.c3]);
            cur.c3++;
        } else if (nqa == 2) {
            q_0 = static_cast<u16>(circ.b0[cur.c2]);
            q_1 = static_cast<u16>(circ.b1[cur.c2]);
            cur.c2++;
        } else if (nqa == 1) {
            q_0 = static_cast<u16>(circ.q0[cur.c1]);
            cur.c1++;
        }
        if (ty & 0x40) b_t = circ.bt[cur.cbt++];
        if (ty & 0x80) b_c = circ.bc[cur.cbc++];
        if (base_op(ty) == T_ZPOW) idx_extra = static_cast<u32>(cur.cang++);
        if (ty == T_TAG) { idx_extra = static_cast<u32>(cur.ctag++); continue; }

        if (ty == T_PUSH) {
            const Word256& cb = st.b[b_c];
            for (int i = 0; i < 4; ++i) bcond.w[i] &= cb.w[i];
            st.cond_stack.push_back(bcond);
            continue;
        }
        if (ty == T_POP) {
            if (!st.cond_stack.empty()) st.cond_stack.pop_back();
            bcond = st.cond_stack.empty() ? ones : st.cond_stack.back();
            continue;
        }

        // Causal-cone fast skip when filter masks are supplied:
        if (cone_q_filter && cone_b_filter) {
            bool relevant = (base_op(ty) == T_NEG);
            if (!relevant && nqa >= 1 && q_0 < cone_q_filter->size() && (*cone_q_filter)[q_0]) relevant = true;
            if (!relevant && nqa >= 2 && q_1 < cone_q_filter->size() && (*cone_q_filter)[q_1]) relevant = true;
            if (!relevant && nqa >= 3 && q_2 < cone_q_filter->size() && (*cone_q_filter)[q_2]) relevant = true;
            if (!relevant && (ty & 0x40) && b_t < cone_b_filter->size() && (*cone_b_filter)[b_t]) relevant = true;
            if (!relevant) continue;
        }
        executed++;

        Word256 cond = bcond;
        bool plain = st.cond_stack.empty();
        if (ty & 0x80) {
            const Word256& cb = st.b[b_c];
            for (int i = 0; i < 4; ++i) cond.w[i] &= cb.w[i];
            plain = false;
        }

        switch (base_op(ty)) {
            case T_X:
                for (int i = 0; i < 4; ++i) st.q[q_0].w[i] ^= cond.w[i];
                break;
            case T_CX:
                for (int i = 0; i < 4; ++i) st.q[q_1].w[i] ^= cond.w[i] & st.q[q_0].w[i];
                break;
            case T_CCX:
                for (int i = 0; i < 4; ++i) st.q[q_2].w[i] ^= cond.w[i] & st.q[q_0].w[i] & st.q[q_1].w[i];
                st.exec_tof += plain ? static_cast<u64>(active_lanes) : popc_active(cond);
                break;
            case T_SWAP:
                for (int i = 0; i < 4; ++i) {
                    u64 d = (st.q[q_0].w[i] ^ st.q[q_1].w[i]) & cond.w[i];
                    st.q[q_0].w[i] ^= d;
                    st.q[q_1].w[i] ^= d;
                }
                break;
            case T_CSWAP:
                for (int i = 0; i < 4; ++i) {
                    u64 d = (st.q[q_1].w[i] ^ st.q[q_2].w[i]) & cond.w[i] & st.q[q_0].w[i];
                    st.q[q_1].w[i] ^= d;
                    st.q[q_2].w[i] ^= d;
                }
                st.exec_tof += plain ? static_cast<u64>(active_lanes) : popc_active(cond);
                break;
            case T_Z:
                for (int i = 0; i < 4; ++i) st.sign_ph.w[i] ^= cond.w[i] & st.q[q_0].w[i];
                break;
            case T_CZ:
                for (int i = 0; i < 4; ++i) st.sign_ph.w[i] ^= cond.w[i] & st.q[q_0].w[i] & st.q[q_1].w[i];
                break;
            case T_CCZ:
                for (int i = 0; i < 4; ++i) st.sign_ph.w[i] ^= cond.w[i] & st.q[q_0].w[i] & st.q[q_1].w[i] & st.q[q_2].w[i];
                st.exec_tof += plain ? static_cast<u64>(active_lanes) : popc_active(cond);
                break;
            case T_NEG:
                for (int i = 0; i < 4; ++i) st.sign_ph.w[i] ^= cond.w[i];
                break;
            case T_R:
                for (int i = 0; i < 4; ++i) {
                    u64 r = (outcomes_mode == 0) ? st.next_rnd() : (outcomes_mode == 1 ? 0ull : ~0ull);
                    st.sign_ph.w[i] ^= st.q[q_0].w[i] & r & cond.w[i];
                    st.q[q_0].w[i] &= ~cond.w[i];
                }
                break;
            case T_HMR:
                for (int i = 0; i < 4; ++i) {
                    u64 r = (outcomes_mode == 0) ? st.next_rnd() : (outcomes_mode == 1 ? 0ull : ~0ull);
                    u64 nb_val = (st.b[b_t].w[i] & ~cond.w[i]) | (r & cond.w[i]);
                    st.sign_ph.w[i] ^= st.q[q_0].w[i] & nb_val & cond.w[i];
                    st.q[q_0].w[i] &= ~cond.w[i];
                    st.b[b_t].w[i] = nb_val;
                }
                break;
            case T_ZPOW: {
                u128 a = circ.angles[idx_extra];
                u8 cl = circ.angle_cls[idx_extra];
                if (cl) {
                    u64 cnt_exec = plain ? static_cast<u64>(active_lanes) : popc_active(cond);
                    if (cl == 1) st.exec_t += cnt_exec;
                    else st.exec_rot += cnt_exec;
                }
                for (int i = 0; i < 4; ++i) {
                    u64 mk = st.q[q_0].w[i] & cond.w[i] & active_mask.w[i];
                    while (mk) {
                        int bit = __builtin_ctzll(mk);
                        mk &= (mk - 1);
                        st.rot_ang[i * 64 + bit] += a;
                    }
                }
                break;
            }
            case T_BST0:
                for (int i = 0; i < 4; ++i) st.b[b_t].w[i] &= ~cond.w[i];
                break;
            case T_BST1:
                for (int i = 0; i < 4; ++i) st.b[b_t].w[i] |= cond.w[i];
                break;
            case T_BINV:
                for (int i = 0; i < 4; ++i) st.b[b_t].w[i] ^= cond.w[i];
                break;
            default:
                break;
        }
    }
    if (ops_executed_out) *ops_executed_out = executed;
}

static void init_random_inputs(SimState256& st, u32 nq, const std::vector<u8>& is_clean, int active_lanes) {
    for (u32 q = 0; q < nq; ++q) {
        if (q < is_clean.size() && is_clean[q]) {
            st.q[q] = Word256{};
        } else {
            for (int i = 0; i < 4; ++i) {
                st.q[q].w[i] = st.next_rnd();
            }
            for (int l = active_lanes; l < 256; ++l) {
                st.q[q].w[l >> 6] &= ~(1ull << (l & 63));
            }
        }
    }
}

// Pack 256-lane words across multiple batches into big-endian uint8 bytes (matching np.packbits(..., bitorder='big')).
static void pack_lanes_big_endian(
    const std::vector<SimState256>& batches,
    u32 wire,
    int total_shots,
    u8* out_row
) {
    size_t row_bytes = static_cast<size_t>((total_shots + 7) / 8);
    std::memset(out_row, 0, row_bytes);
    for (int s = 0; s < total_shots; ++s) {
        int b_idx = s >> 8;
        int lane = s & 255;
        u64 bit = (batches[b_idx].q[wire].w[lane >> 6] >> (lane & 63)) & 1ull;
        if (bit) {
            out_row[s >> 3] |= static_cast<u8>(0x80u >> (s & 7));
        }
    }
}

// ---------------------------------------------------------------------------
// GF(2^64) Arithmetic for Lovasz Linear Matroid Parity
// ---------------------------------------------------------------------------
static inline u64 gf2_64_mul(u64 a, u64 b) {
    // Irreducible polynomial x^64 + x^4 + x^3 + x + 1 (0x1b).
    u64 res = 0;
    while (b) {
        if (b & 1ull) res ^= a;
        u64 hi = a >> 63;
        a <<= 1;
        if (hi) a ^= 0x1bull;
        b >>= 1;
    }
    return res;
}

static u64 gf2_64_pow(u64 a, u64 e) {
    u64 r = 1ull;
    while (e) {
        if (e & 1ull) r = gf2_64_mul(r, a);
        a = gf2_64_mul(a, a);
        e >>= 1;
    }
    return r;
}

static inline u64 gf2_64_inv(u64 a) {
    // a^(2^64 - 2) in GF(2^64)
    return a == 0 ? 0 : gf2_64_pow(a, ~1ull);
}

// ---------------------------------------------------------------------------
// Python C-API Types: KmbIndexObject & BitSlicedEngineObject
// ---------------------------------------------------------------------------
struct KmbIndexObject {
    PyObject_HEAD
    std::shared_ptr<ColumnarCircuit> circ;
};

static void KmbIndex_dealloc(KmbIndexObject* self) {
    self->circ.~shared_ptr<ColumnarCircuit>();
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

static PyObject* KmbIndex_new(PyTypeObject* type, PyObject*, PyObject*) {
    auto* self = reinterpret_cast<KmbIndexObject*>(type->tp_alloc(type, 0));
    if (self) {
        new (&self->circ) std::shared_ptr<ColumnarCircuit>();
    }
    return reinterpret_cast<PyObject*>(self);
}

static int KmbIndex_init(KmbIndexObject* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"path", "kmx_text", nullptr};
    const char* path = nullptr;
    const char* kmx_text = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|zz", const_cast<char**>(kwlist), &path, &kmx_text)) {
        return -1;
    }
    try {
        if (path && kmx_text) {
            PyErr_SetString(PyExc_ValueError, "Supply either path or kmx_text, not both");
            return -1;
        }
        if (path) {
            self->circ = ColumnarCircuit::from_file(path);
        } else if (kmx_text) {
            self->circ = ColumnarCircuit::from_kmx_text(kmx_text);
        } else {
            PyErr_SetString(PyExc_ValueError, "KmbIndex requires path or kmx_text");
            return -1;
        }
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }
    return 0;
}

static Py_ssize_t KmbIndex_len(KmbIndexObject* self) {
    return static_cast<Py_ssize_t>(self->circ->num_ops);
}

static PyObject* KmbIndex_info(KmbIndexObject* self, PyObject*) {
    const auto& c = *self->circ;
    PyObject* d = PyDict_New();
    PyDict_SetItemString(d, "path", PyUnicode_FromString(c.path.c_str()));
    PyDict_SetItemString(d, "sha256", PyUnicode_FromString(c.sha256.c_str()));
    PyDict_SetItemString(d, "file_bytes", PyLong_FromUnsignedLongLong(c.file_bytes));
    PyDict_SetItemString(d, "is_kmb", PyBool_FromLong(c.is_kmb ? 1 : 0));
    PyDict_SetItemString(d, "num_qubits", PyLong_FromUnsignedLong(c.nq));
    PyDict_SetItemString(d, "num_bits", PyLong_FromUnsignedLong(c.nb));
    PyDict_SetItemString(d, "num_registers", PyLong_FromSize_t(c.regs.size()));
    PyDict_SetItemString(d, "num_ops", PyLong_FromUnsignedLongLong(c.num_ops));
    PyDict_SetItemString(d, "toffoli_static", PyLong_FromUnsignedLongLong(c.static_toffoli));
    PyDict_SetItemString(d, "t_static", PyLong_FromUnsignedLongLong(c.static_t));
    PyDict_SetItemString(d, "rotations_static", PyLong_FromUnsignedLongLong(c.static_rot));
    PyDict_SetItemString(d, "open_ms", PyFloat_FromDouble(c.open_ms));

    PyObject* hist = PyDict_New();
    for (int ty = 0; ty < 256; ++ty) {
        if (c.opcode_hist[ty] > 0) {
            std::string name = op_mnemonic(static_cast<u8>(ty));
            if ((ty & 0x80) && ty != T_PUSH) name += "_IF";
            PyObject* prev = PyDict_GetItemString(hist, name.c_str());
            u64 v = c.opcode_hist[ty] + (prev ? PyLong_AsUnsignedLongLong(prev) : 0ull);
            PyObject* val = PyLong_FromUnsignedLongLong(v);
            PyDict_SetItemString(hist, name.c_str(), val);
            Py_DECREF(val);
        }
    }
    PyDict_SetItemString(d, "census", hist);
    Py_DECREF(hist);

    PyObject* regs = PyDict_New();
    for (const auto& r : c.regs) {
        PyObject* lst = PyList_New(static_cast<Py_ssize_t>(r.second.size()));
        for (size_t i = 0; i < r.second.size(); ++i) {
            u32 raw_id = r.second[i];
            std::string label = (raw_id & 0x80000000u)
                ? ("q" + std::to_string(raw_id & 0x7fffffffu))
                : ("b" + std::to_string(raw_id));
            PyList_SET_ITEM(lst, static_cast<Py_ssize_t>(i), PyUnicode_FromString(label.c_str()));
        }
        PyDict_SetItemString(regs, r.first.c_str(), lst);
        Py_DECREF(lst);
    }
    PyDict_SetItemString(d, "registers", regs);
    Py_DECREF(regs);
    return d;
}

static PyObject* KmbIndex_decode_span(KmbIndexObject* self, PyObject* args) {
    unsigned long long start = 0, stop = 0;
    if (!PyArg_ParseTuple(args, "KK", &start, &stop)) return nullptr;
    try {
        std::vector<OpRec> ops;
        self->circ->extract_ops(start, stop, ops);
        PyObject* lst = PyList_New(static_cast<Py_ssize_t>(ops.size()));
        for (size_t i = 0; i < ops.size(); ++i) {
            std::string s = self->circ->format_op(ops[i]);
            PyList_SET_ITEM(lst, static_cast<Py_ssize_t>(i), PyUnicode_FromString(s.c_str()));
        }
        return lst;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

static PyObject* KmbIndex_fragment_sha256(KmbIndexObject* self, PyObject* args) {
    unsigned long long start = 0, stop = 0;
    if (!PyArg_ParseTuple(args, "KK", &start, &stop)) return nullptr;
    try {
        std::string h = self->circ->fragment_sha256(start, stop);
        return PyUnicode_FromString(h.c_str());
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

static PyTypeObject KmbIndexType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
};

static PyObject* KmbIndex_same_span(KmbIndexObject* self, PyObject* args) {
    unsigned long long s1 = 0, e1 = 0, s2 = 0, e2 = 0;
    PyObject* other_obj = nullptr;
    if (!PyArg_ParseTuple(args, "KKOKK", &s1, &e1, &other_obj, &s2, &e2)) return nullptr;
    if (!PyObject_TypeCheck(other_obj, &KmbIndexType)) {
        PyErr_SetString(PyExc_TypeError, "Expected KmbIndex for other");
        return nullptr;
    }
    try {
        auto* other = reinterpret_cast<KmbIndexObject*>(other_obj);
        bool eq = self->circ->same_span(s1, e1, *other->circ, s2, e2);
        return PyBool_FromLong(eq ? 1 : 0);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

static PyObject* KmbIndex_causal_cone(KmbIndexObject* self, PyObject* args) {
    unsigned long long start = 0, stop = 0;
    if (!PyArg_ParseTuple(args, "KK", &start, &stop)) return nullptr;
    try {
        const auto& c = *self->circ;
        std::vector<OpRec> ops;
        c.extract_ops(start, stop, ops);
        std::vector<u8> q_touch(c.nq, 0), q_write(c.nq, 0), b_touch(c.nb, 0);
        bool has_hmr = false, has_zpow = false;
        std::vector<u32> cond_stack;
        BlockIdx cur = c.seek_cursor(start);
        (void)cur;
        for (const auto& o : ops) {
            u8 ty = o.ty;
            u8 b = base_op(ty);
            if (ty == T_PUSH) {
                if (o.bc < b_touch.size()) b_touch[o.bc] = 1;
                cond_stack.push_back(o.bc);
                continue;
            }
            if (ty == T_POP) {
                if (!cond_stack.empty()) cond_stack.pop_back();
                continue;
            }
            if (ty == T_TAG) continue;
            int nqa = nqarg(ty);
            if (nqa >= 1 && o.q0 < q_touch.size()) q_touch[o.q0] = 1;
            if (nqa >= 2 && o.q1 < q_touch.size()) q_touch[o.q1] = 1;
            if (nqa >= 3 && o.q2 < q_touch.size()) q_touch[o.q2] = 1;
            if (b == T_X || b == T_R || b == T_HMR) { if (o.q0 < q_write.size()) q_write[o.q0] = 1; }
            else if (b == T_CX) { if (o.q1 < q_write.size()) q_write[o.q1] = 1; }
            else if (b == T_CCX) { if (o.q2 < q_write.size()) q_write[o.q2] = 1; }
            else if (b == T_SWAP) { if (o.q0 < q_write.size()) q_write[o.q0] = 1; if (o.q1 < q_write.size()) q_write[o.q1] = 1; }
            else if (b == T_CSWAP) { if (o.q1 < q_write.size()) q_write[o.q1] = 1; if (o.q2 < q_write.size()) q_write[o.q2] = 1; }
            if ((ty & 0x40) && o.bt < b_touch.size()) b_touch[o.bt] = 1;
            if ((ty & 0x80) && o.bc < b_touch.size()) b_touch[o.bc] = 1;
            if (b == T_HMR) has_hmr = true;
            if (b == T_ZPOW) has_zpow = true;
        }
        auto vec_to_pylist = [](const std::vector<u8>& mask) {
            PyObject* lst = PyList_New(0);
            for (size_t i = 0; i < mask.size(); ++i) {
                if (mask[i]) {
                    PyObject* item = PyLong_FromSize_t(i);
                    PyList_Append(lst, item);
                    Py_DECREF(item);
                }
            }
            return lst;
        };
        PyObject* res = PyDict_New();
        PyObject* ql = vec_to_pylist(q_touch);
        PyObject* qw = vec_to_pylist(q_write);
        PyObject* bl = vec_to_pylist(b_touch);
        PyDict_SetItemString(res, "qubits", ql); Py_DECREF(ql);
        PyDict_SetItemString(res, "written_qubits", qw); Py_DECREF(qw);
        PyDict_SetItemString(res, "bits", bl); Py_DECREF(bl);
        PyDict_SetItemString(res, "has_hmr", PyBool_FromLong(has_hmr ? 1 : 0));
        PyDict_SetItemString(res, "has_z_pow", PyBool_FromLong(has_zpow ? 1 : 0));
        return res;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

static PyObject* KmbIndex_compose_kmb(KmbIndexObject* self, PyObject* args) {
    PyObject* edits_obj = nullptr;
    PyObject* donors_obj = nullptr;
    const char* out_path = nullptr;
    if (!PyArg_ParseTuple(args, "OOs", &edits_obj, &donors_obj, &out_path)) return nullptr;
    try {
        auto t0 = std::chrono::steady_clock::now();
        if (!PySequence_Check(donors_obj) || !PySequence_Check(edits_obj)) {
            throw std::runtime_error("edits and donors must be sequences");
        }
        std::vector<const ColumnarCircuit*> donors;
        Py_ssize_t nd = PySequence_Size(donors_obj);
        for (Py_ssize_t i = 0; i < nd; ++i) {
            PyObject* item = PySequence_GetItem(donors_obj, i);
            if (!PyObject_TypeCheck(item, &KmbIndexType)) {
                Py_DECREF(item);
                throw std::runtime_error("All donors must be KmbIndex objects");
            }
            donors.push_back(reinterpret_cast<KmbIndexObject*>(item)->circ.get());
            Py_DECREF(item);
        }

        const ColumnarCircuit& parent = *self->circ;
        std::vector<SpanSeg> segs;
        u64 cursor = 0;
        Py_ssize_t ne = PySequence_Size(edits_obj);
        for (Py_ssize_t i = 0; i < ne; ++i) {
            PyObject* tup = PySequence_GetItem(edits_obj, i);
            unsigned long long ps = 0, pe = 0, ds = 0, de = 0;
            Py_ssize_t didx = 0;
            if (!PyArg_ParseTuple(tup, "KKnKK", &ps, &pe, &didx, &ds, &de)) {
                Py_DECREF(tup);
                return nullptr;
            }
            Py_DECREF(tup);
            if (didx < 0 || static_cast<size_t>(didx) >= donors.size()) {
                throw std::runtime_error("Donor index out of bounds");
            }
            if (ps < cursor || pe < ps || pe > parent.num_ops) {
                throw std::runtime_error("Invalid or unsorted parent interval in compose_kmb");
            }
            if (ps > cursor) {
                segs.push_back(SpanSeg{&parent, cursor, ps, parent.seek_cursor(cursor), parent.seek_cursor(ps)});
            }
            const ColumnarCircuit* donor = donors[static_cast<size_t>(didx)];
            if (de < ds || de > donor->num_ops) {
                throw std::runtime_error("Invalid donor interval in compose_kmb");
            }
            if (de > ds) {
                segs.push_back(SpanSeg{donor, ds, de, donor->seek_cursor(ds), donor->seek_cursor(de)});
            }
            cursor = pe;
        }
        if (cursor < parent.num_ops) {
            segs.push_back(SpanSeg{&parent, cursor, parent.num_ops, parent.seek_cursor(cursor), parent.seek_cursor(parent.num_ops)});
        }

        std::string sha = write_composed_kmb(parent, segs, out_path);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        u64 total_ops = 0;
        for (const auto& sg : segs) total_ops += (sg.stop - sg.start);

        PyObject* res = PyDict_New();
        PyDict_SetItemString(res, "result_sha256", PyUnicode_FromString(sha.c_str()));
        PyDict_SetItemString(res, "result_operations", PyLong_FromUnsignedLongLong(total_ops));
        PyDict_SetItemString(res, "elapsed_ms", PyFloat_FromDouble(ms));
        return res;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

static PyMethodDef KmbIndex_methods[] = {
    {"info", reinterpret_cast<PyCFunction>(KmbIndex_info), METH_NOARGS, "Return zero-copy metadata and opcode census."},
    {"decode_span", reinterpret_cast<PyCFunction>(KmbIndex_decode_span), METH_VARARGS, "Decode [start, stop) to canonical KMX lines."},
    {"fragment_sha256", reinterpret_cast<PyCFunction>(KmbIndex_fragment_sha256), METH_VARARGS, "SHA-256 of canonical UTF-8 lines in [start, stop)."},
    {"same_span", reinterpret_cast<PyCFunction>(KmbIndex_same_span), METH_VARARGS, "Fast columnar span equality check."},
    {"causal_cone", reinterpret_cast<PyCFunction>(KmbIndex_causal_cone), METH_VARARGS, "Compute touched/written qubits and bits in [start, stop)."},
    {"compose_kmb", reinterpret_cast<PyCFunction>(KmbIndex_compose_kmb), METH_VARARGS, "Compose binary .kmb intervals directly without text spooling."},
    {nullptr, nullptr, 0, nullptr}
};

static PySequenceMethods KmbIndex_seq = {
    reinterpret_cast<lenfunc>(KmbIndex_len),
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr
};

// ---------------------------------------------------------------------------
// BitSlicedEngine Python Object (Hot Checkpointed Workspace)
// ---------------------------------------------------------------------------
struct BitSlicedEngineObject {
    PyObject_HEAD
    std::shared_ptr<ColumnarCircuit> circ;
    u64 checkpoint_stride;
    u64 cached_seed;
    std::vector<u8> cached_clean;
    std::map<u64, SimState256> checkpoints;
};

static void Engine_dealloc(BitSlicedEngineObject* self) {
    self->circ.~shared_ptr<ColumnarCircuit>();
    self->cached_clean.~vector<u8>();
    self->checkpoints.~map<u64, SimState256>();
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

static PyObject* Engine_new(PyTypeObject* type, PyObject*, PyObject*) {
    auto* self = reinterpret_cast<BitSlicedEngineObject*>(type->tp_alloc(type, 0));
    if (self) {
        new (&self->circ) std::shared_ptr<ColumnarCircuit>();
        new (&self->cached_clean) std::vector<u8>();
        new (&self->checkpoints) std::map<u64, SimState256>();
        self->checkpoint_stride = 8192;
        self->cached_seed = 0;
    }
    return reinterpret_cast<PyObject*>(self);
}

static int Engine_init(BitSlicedEngineObject* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"index", "checkpoint_stride", nullptr};
    PyObject* idx_obj = nullptr;
    unsigned long long stride = 8192;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|K", const_cast<char**>(kwlist), &idx_obj, &stride)) {
        return -1;
    }
    if (!PyObject_TypeCheck(idx_obj, &KmbIndexType)) {
        PyErr_SetString(PyExc_TypeError, "BitSlicedEngine requires a KmbIndex");
        return -1;
    }
    self->circ = reinterpret_cast<KmbIndexObject*>(idx_obj)->circ;
    self->checkpoint_stride = std::max<u64>(256, stride);
    self->checkpoints.clear();
    return 0;
}

static std::vector<u8> parse_clean_wires(PyObject* clean_obj, u32 nq) {
    std::vector<u8> clean(nq, 0);
    if (!clean_obj || clean_obj == Py_None) return clean;
    PyObject* seq = PySequence_Fast(clean_obj, "clean_wires must be a sequence");
    if (!seq) throw std::runtime_error("clean_wires must be a sequence");
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        long w = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i));
        if (w >= 0 && static_cast<u32>(w) < nq) clean[static_cast<size_t>(w)] = 1;
    }
    Py_DECREF(seq);
    return clean;
}

static SimState256& get_or_build_checkpoint(
    BitSlicedEngineObject* self,
    u64 target_op,
    u64 seed,
    const std::vector<u8>& clean
) {
    if (self->cached_seed != seed || self->cached_clean != clean || self->checkpoints.empty()) {
        self->checkpoints.clear();
        self->cached_seed = seed;
        self->cached_clean = clean;
        SimState256 init_st;
        init_st.init(self->circ->nq, self->circ->nb, seed);
        init_random_inputs(init_st, self->circ->nq, clean, 256);
        self->checkpoints[0] = std::move(init_st);
    }
    u64 aligned = (target_op / self->checkpoint_stride) * self->checkpoint_stride;
    auto it = self->checkpoints.upper_bound(aligned);
    --it;
    u64 pos = it->first;
    SimState256 cur = it->second;
    while (pos + self->checkpoint_stride <= aligned) {
        u64 next_pos = pos + self->checkpoint_stride;
        run_span_256(*self->circ, pos, next_pos, cur, 0, 256);
        pos = next_pos;
        self->checkpoints[pos] = cur;
    }
    return self->checkpoints[pos];
}

static PyObject* Engine_trace_cuts(BitSlicedEngineObject* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"cuts", "shots", "seed", "clean_wires", nullptr};
    PyObject* cuts_obj = nullptr;
    int shots = 256;
    unsigned long long seed = 1;
    PyObject* clean_obj = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iKO", const_cast<char**>(kwlist), &cuts_obj, &shots, &seed, &clean_obj)) {
        return nullptr;
    }
    try {
        auto t0 = std::chrono::steady_clock::now();
        if (shots < 1 || shots > 65536) throw std::runtime_error("shots must be in 1..65536");
        const auto& c = *self->circ;
        std::vector<u8> clean = parse_clean_wires(clean_obj, c.nq);

        PyObject* seq = PySequence_Fast(cuts_obj, "cuts must be a sequence");
        if (!seq) return nullptr;
        Py_ssize_t ncuts = PySequence_Fast_GET_SIZE(seq);
        std::vector<u64> cuts(static_cast<size_t>(ncuts));
        for (Py_ssize_t i = 0; i < ncuts; ++i) {
            long long v = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(seq, i));
            if (v < 0 || static_cast<u64>(v) > c.num_ops) {
                Py_DECREF(seq);
                throw std::runtime_error("cut position out of range");
            }
            cuts[static_cast<size_t>(i)] = static_cast<u64>(v);
        }
        Py_DECREF(seq);
        if (!std::is_sorted(cuts.begin(), cuts.end())) {
            throw std::runtime_error("cuts must be sorted in nondecreasing order");
        }

        int nbatches = (shots + 255) / 256;
        std::vector<SimState256> states(static_cast<size_t>(nbatches));
        for (int b = 0; b < nbatches; ++b) {
            int lanes = std::min(256, shots - b * 256);
            states[b].init(c.nq, c.nb, seed + static_cast<u64>(b) * 1000003ull);
            init_random_inputs(states[b], c.nq, clean, lanes);
        }

        size_t row_bytes = static_cast<size_t>((shots + 7) / 8);
        PyObject* cut_list = PyList_New(ncuts);
        u64 prev = 0;
        for (size_t ci = 0; ci < cuts.size(); ++ci) {
            u64 target = cuts[ci];
            if (target > prev) {
                #pragma omp parallel for schedule(static) if(nbatches > 1)
                for (int b = 0; b < nbatches; ++b) {
                    int lanes = std::min(256, shots - b * 256);
                    run_span_256(c, prev, target, states[b], 0, lanes);
                }
                prev = target;
            }
            std::vector<u8> packed_q(static_cast<size_t>(c.nq) * row_bytes, 0);
            for (u32 q = 0; q < c.nq; ++q) {
                pack_lanes_big_endian(states, q, shots, packed_q.data() + static_cast<size_t>(q) * row_bytes);
            }
            // Also extract Z8 phase residues (0..7 in units of pi/4, i.e., top 3 bits of total_phase)
            std::vector<u8> z8_phase(static_cast<size_t>(shots), 0);
            for (int s = 0; s < shots; ++s) {
                u128 ph = states[s >> 8].total_phase(s & 255);
                // Round/shift top 3 bits (units of 1/8 turn = pi/4 radians)
                z8_phase[static_cast<size_t>(s)] = static_cast<u8>((ph >> 125) & 7u);
            }

            PyObject* entry = PyDict_New();
            PyDict_SetItemString(entry, "cut", PyLong_FromUnsignedLongLong(target));
            PyObject* q_bytes = PyBytes_FromStringAndSize(reinterpret_cast<const char*>(packed_q.data()), static_cast<Py_ssize_t>(packed_q.size()));
            PyObject* ph_bytes = PyBytes_FromStringAndSize(reinterpret_cast<const char*>(z8_phase.data()), static_cast<Py_ssize_t>(z8_phase.size()));
            PyDict_SetItemString(entry, "packed_qubits", q_bytes); Py_DECREF(q_bytes);
            PyDict_SetItemString(entry, "z8_phases", ph_bytes); Py_DECREF(ph_bytes);
            PyList_SET_ITEM(cut_list, static_cast<Py_ssize_t>(ci), entry);
        }

        u64 total_tof = 0, total_t = 0;
        for (int b = 0; b < nbatches; ++b) {
            total_tof += states[b].exec_tof;
            total_t += states[b].exec_t;
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        PyObject* out = PyDict_New();
        PyDict_SetItemString(out, "shots", PyLong_FromLong(shots));
        PyDict_SetItemString(out, "num_qubits", PyLong_FromUnsignedLong(c.nq));
        PyDict_SetItemString(out, "row_bytes", PyLong_FromSize_t(row_bytes));
        PyDict_SetItemString(out, "cuts", cut_list); Py_DECREF(cut_list);
        PyDict_SetItemString(out, "mean_executed_toffolis", PyFloat_FromDouble(static_cast<double>(total_tof) / shots));
        PyDict_SetItemString(out, "mean_executed_t", PyFloat_FromDouble(static_cast<double>(total_t) / shots));
        PyDict_SetItemString(out, "elapsed_ms", PyFloat_FromDouble(ms));
        return out;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

static PyObject* Engine_verify_cone(BitSlicedEngineObject* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"start", "stop", "replacement", "seed", "clean_wires", "outcomes_mode", nullptr};
    unsigned long long start = 0, stop = 0;
    PyObject* rep_obj = nullptr;
    unsigned long long seed = 1;
    PyObject* clean_obj = nullptr;
    int outcomes_mode = 1;  // deterministic 0-outcome mode by default for exact value+phase cone diff, or 0 for matched RNG
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "KKO|KOi", const_cast<char**>(kwlist), &start, &stop, &rep_obj, &seed, &clean_obj, &outcomes_mode)) {
        return nullptr;
    }
    try {
        auto t0 = std::chrono::steady_clock::now();
        const auto& parent = *self->circ;
        if (start > stop || stop > parent.num_ops) throw std::runtime_error("Invalid [start, stop) interval");

        std::shared_ptr<ColumnarCircuit> rep;
        if (PyObject_TypeCheck(rep_obj, &KmbIndexType)) {
            rep = reinterpret_cast<KmbIndexObject*>(rep_obj)->circ;
        } else if (PyUnicode_Check(rep_obj)) {
            const char* txt = PyUnicode_AsUTF8(rep_obj);
            rep = ColumnarCircuit::from_kmx_text(txt ? txt : "");
        } else {
            throw std::runtime_error("replacement must be a KmbIndex or KMX string");
        }

        u32 max_nq = std::max(parent.nq, rep->nq);
        u32 max_nb = std::max(parent.nb, rep->nb);
        std::vector<u8> clean = parse_clean_wires(clean_obj, max_nq);

        // Retrieve nearest cached checkpoint <= start
        std::vector<u8> parent_clean(clean.begin(), clean.begin() + parent.nq);
        SimState256& cp = get_or_build_checkpoint(self, start, seed, parent_clean);
        u64 cp_op = (start / self->checkpoint_stride) * self->checkpoint_stride;
        SimState256 base_state = cp;
        if (start > cp_op) {
            run_span_256(parent, cp_op, start, base_state, 0, 256);
        }
        base_state.q.resize(max_nq, Word256{});
        base_state.b.resize(max_nb, Word256{});

        // Compute union causal cone of parent[start:stop] and replacement[0:rep->num_ops]
        std::vector<u8> cone_q(max_nq, 0), cone_b(max_nb, 0);
        auto mark_cone = [&](const ColumnarCircuit& cc, u64 s, u64 e) {
            std::vector<OpRec> ops;
            cc.extract_ops(s, e, ops);
            for (const auto& o : ops) {
                if (o.ty == T_TAG) continue;
                int nqa = nqarg(o.ty);
                if (nqa >= 1 && o.q0 < max_nq) cone_q[o.q0] = 1;
                if (nqa >= 2 && o.q1 < max_nq) cone_q[o.q1] = 1;
                if (nqa >= 3 && o.q2 < max_nq) cone_q[o.q2] = 1;
                if ((o.ty & 0x40) && o.bt < max_nb) cone_b[o.bt] = 1;
                if ((o.ty & 0x80) && o.bc < max_nb) cone_b[o.bc] = 1;
            }
        };
        mark_cone(parent, start, stop);
        mark_cone(*rep, 0, rep->num_ops);

        // Run both across deterministic outcome modes {1 (all-0), 2 (all-1)} when outcomes_mode == -1, or specified mode
        std::vector<int> modes = (outcomes_mode < 0) ? std::vector<int>{1, 2, 0} : std::vector<int>{outcomes_mode};
        bool passed = true;
        int bad_lane = -1;
        std::string fail_reason;
        u64 cone_ops_exec = 0;

        for (int om : modes) {
            SimState256 st_p = base_state;
            SimState256 st_c = base_state;
            // Re-seed identically for matched random draws if om == 0
            st_p.rng[0] = st_c.rng[0] = seed ^ 0x123456789abcdef0ull;
            u64 ex1 = 0, ex2 = 0;
            run_span_256(parent, start, stop, st_p, om, 256, &cone_q, &cone_b, &ex1);
            run_span_256(*rep, 0, rep->num_ops, st_c, om, 256, &cone_q, &cone_b, &ex2);
            cone_ops_exec = ex1 + ex2;

            for (u32 q = 0; q < max_nq; ++q) {
                if (!cone_q[q]) continue;
                for (int i = 0; i < 4; ++i) {
                    u64 diff = st_p.q[q].w[i] ^ st_c.q[q].w[i];
                    if (diff) {
                        passed = false;
                        bad_lane = i * 64 + __builtin_ctzll(diff);
                        fail_reason = "value_mismatch_q" + std::to_string(q);
                        break;
                    }
                }
                if (!passed) break;
            }
            if (!passed) break;

            // Compare relative phase across lanes (global constant phase allowed only if 0; Kickmix checks exact per-shot phase)
            for (int l = 0; l < 256; ++l) {
                if (st_p.total_phase(l) != st_c.total_phase(l)) {
                    passed = false;
                    bad_lane = l;
                    fail_reason = "phase_mismatch";
                    break;
                }
            }
            if (!passed) break;
        }

        size_t cone_q_cnt = 0;
        for (u8 f : cone_q) if (f) cone_q_cnt++;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        PyObject* res = PyDict_New();
        PyDict_SetItemString(res, "passed", PyBool_FromLong(passed ? 1 : 0));
        PyDict_SetItemString(res, "checkpoint_op", PyLong_FromUnsignedLongLong(cp_op));
        PyDict_SetItemString(res, "cone_qubits", PyLong_FromSize_t(cone_q_cnt));
        PyDict_SetItemString(res, "cone_ops_executed", PyLong_FromUnsignedLongLong(cone_ops_exec));
        PyDict_SetItemString(res, "counterexample_lane", bad_lane >= 0 ? PyLong_FromLong(bad_lane) : Py_NewRef(Py_None));
        PyDict_SetItemString(res, "reason", PyUnicode_FromString(passed ? "verified_cone_256" : fail_reason.c_str()));
        PyDict_SetItemString(res, "elapsed_ms", PyFloat_FromDouble(ms));
        return res;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

static PyObject* Engine_verify_cones_batch(BitSlicedEngineObject* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"start", "stop", "replacements", "seed", "clean_wires", nullptr};
    unsigned long long start = 0, stop = 0;
    PyObject* reps_obj = nullptr;
    unsigned long long seed = 1;
    PyObject* clean_obj = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "KKO|KO", const_cast<char**>(kwlist), &start, &stop, &reps_obj, &seed, &clean_obj)) {
        return nullptr;
    }
    try {
        auto t0 = std::chrono::steady_clock::now();
        const auto& parent = *self->circ;
        PyObject* seq = PySequence_Fast(reps_obj, "replacements must be a sequence");
        if (!seq) return nullptr;
        Py_ssize_t M = PySequence_Fast_GET_SIZE(seq);
        std::vector<std::shared_ptr<ColumnarCircuit>> reps(static_cast<size_t>(M));
        u32 max_nq = parent.nq, max_nb = parent.nb;
        for (Py_ssize_t i = 0; i < M; ++i) {
            PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
            if (PyObject_TypeCheck(item, &KmbIndexType)) {
                reps[static_cast<size_t>(i)] = reinterpret_cast<KmbIndexObject*>(item)->circ;
            } else if (PyUnicode_Check(item)) {
                const char* txt = PyUnicode_AsUTF8(item);
                reps[static_cast<size_t>(i)] = ColumnarCircuit::from_kmx_text(txt ? txt : "");
            } else {
                Py_DECREF(seq);
                throw std::runtime_error("Each replacement must be a KmbIndex or KMX string");
            }
            max_nq = std::max(max_nq, reps[static_cast<size_t>(i)]->nq);
            max_nb = std::max(max_nb, reps[static_cast<size_t>(i)]->nb);
        }
        Py_DECREF(seq);

        std::vector<u8> clean = parse_clean_wires(clean_obj, max_nq);
        std::vector<u8> parent_clean(clean.begin(), clean.begin() + parent.nq);
        SimState256& cp = get_or_build_checkpoint(self, start, seed, parent_clean);
        u64 cp_op = (start / self->checkpoint_stride) * self->checkpoint_stride;
        SimState256 base_state = cp;
        if (start > cp_op) run_span_256(parent, cp_op, start, base_state, 0, 256);
        base_state.q.resize(max_nq, Word256{});
        base_state.b.resize(max_nb, Word256{});

        // Advance parent reference once across modes {1, 2}
        SimState256 ref_m1 = base_state, ref_m2 = base_state;
        run_span_256(parent, start, stop, ref_m1, 1, 256);
        run_span_256(parent, start, stop, ref_m2, 2, 256);

        long long parent_span_tof = 0;
        for (u64 op = start; op < stop; ++op) {
            u8 b = parent.types[op] & 0x3f;
            if (b == T_CCX || b == T_CSWAP || b == T_CCZ) parent_span_tof++;
        }

        std::vector<u8> verdicts(static_cast<size_t>(M), 0);
        std::vector<long long> tof_deltas(static_cast<size_t>(M), 0);

        #pragma omp parallel for schedule(dynamic) if(M > 2)
        for (Py_ssize_t idx = 0; idx < M; ++idx) {
            const auto& r = *reps[static_cast<size_t>(idx)];
            bool ok = true;
            for (int pass = 0; pass < 2 && ok; ++pass) {
                int om = (pass == 0) ? 1 : 2;
                const SimState256& ref = (pass == 0) ? ref_m1 : ref_m2;
                SimState256 cand = base_state;
                run_span_256(r, 0, r.num_ops, cand, om, 256);
                if (pass == 0) {
                    tof_deltas[static_cast<size_t>(idx)] =
                        static_cast<long long>(r.static_toffoli) - parent_span_tof;
                }
                for (u32 q = 0; q < max_nq && ok; ++q) {
                    for (int w = 0; w < 4; ++w) {
                        if (ref.q[q].w[w] != cand.q[q].w[w]) { ok = false; break; }
                    }
                }
                for (int l = 0; l < 256 && ok; ++l) {
                    if (ref.total_phase(l) != cand.total_phase(l)) { ok = false; break; }
                }
            }
            verdicts[static_cast<size_t>(idx)] = ok ? 1 : 0;
        }

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        PyObject* vlist = PyList_New(M);
        for (Py_ssize_t i = 0; i < M; ++i) {
            PyList_SET_ITEM(vlist, i, PyBool_FromLong(verdicts[static_cast<size_t>(i)]));
        }
        PyObject* out = PyDict_New();
        PyDict_SetItemString(out, "passed", vlist); Py_DECREF(vlist);
        PyDict_SetItemString(out, "candidates_tested", PyLong_FromSsize_t(M));
        PyDict_SetItemString(out, "elapsed_ms", PyFloat_FromDouble(ms));
        PyDict_SetItemString(out, "per_candidate_ms", PyFloat_FromDouble(M > 0 ? ms / M : 0.0));
        return out;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

static PyObject* Engine_profile_liveness(BitSlicedEngineObject* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"seed", "clean_wires", nullptr};
    unsigned long long seed = 1;
    PyObject* clean_obj = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|KO", const_cast<char**>(kwlist), &seed, &clean_obj)) {
        return nullptr;
    }
    try {
        auto t0 = std::chrono::steady_clock::now();
        const auto& c = *self->circ;
        std::vector<u8> clean = parse_clean_wires(clean_obj, c.nq);

        // Single pass R55 proved-zero + classical-prefix + HMR count + peak semantic window
        std::vector<u8> proved_zero = clean;
        u64 classical_prefix_ops = 0;
        bool in_classical_prefix = true;
        u64 hmr_count = 0;
        u32 peak_live = 0;
        u64 first_peak_op = 0, last_peak_op = 0;
        auto count_nonzero = [&]() {
            u32 cnt = 0;
            for (u8 z : proved_zero) if (!z) cnt++;
            return cnt;
        };
        peak_live = count_nonzero();

        BlockIdx cur{};
        for (u64 k = 0; k < c.num_ops; ++k) {
            u8 ty = c.types[k];
            u8 b = base_op(ty);
            int nqa = nqarg(ty);
            u16 q_0 = 0, q_1 = 0, q_2 = 0;
            if (nqa == 3) {
                q_0 = static_cast<u16>(c.a0[cur.c3]);
                q_1 = static_cast<u16>(c.a1[cur.c3]);
                q_2 = static_cast<u16>(c.a2[cur.c3]);
                cur.c3++;
            } else if (nqa == 2) {
                q_0 = static_cast<u16>(c.b0[cur.c2]);
                q_1 = static_cast<u16>(c.b1[cur.c2]);
                cur.c2++;
            } else if (nqa == 1) {
                q_0 = static_cast<u16>(c.q0[cur.c1]);
                cur.c1++;
            }
            if (ty & 0x40) cur.cbt++;
            if (ty & 0x80) cur.cbc++;
            if (b == T_ZPOW) cur.cang++;
            if (ty == T_TAG) { cur.ctag++; continue; }
            if (ty == T_PUSH) { cur.depth++; if (in_classical_prefix) classical_prefix_ops = k + 1; continue; }
            if (ty == T_POP) { if (cur.depth > 0) cur.depth--; if (in_classical_prefix) classical_prefix_ops = k + 1; continue; }

            if (in_classical_prefix) {
                if (b == T_BST0 || b == T_BST1 || b == T_BINV) {
                    classical_prefix_ops = k + 1;
                } else {
                    in_classical_prefix = false;
                }
            }

            bool uncond = !(ty & 0x80) && (cur.depth == 0);
            if (b == T_HMR) {
                hmr_count++;
                if (uncond && q_0 < c.nq) proved_zero[q_0] = 1;
            } else if (b == T_R) {
                if (uncond && q_0 < c.nq) proved_zero[q_0] = 1;
            } else if (b == T_X) {
                if (q_0 < c.nq) proved_zero[q_0] = 0;
            } else if (b == T_CX) {
                if (q_0 < c.nq && q_1 < c.nq && !proved_zero[q_0]) proved_zero[q_1] = 0;
            } else if (b == T_CCX) {
                if (q_0 < c.nq && q_1 < c.nq && q_2 < c.nq && !proved_zero[q_0] && !proved_zero[q_1]) {
                    proved_zero[q_2] = 0;
                }
            } else if (b == T_SWAP) {
                if (uncond && q_0 < c.nq && q_1 < c.nq) std::swap(proved_zero[q_0], proved_zero[q_1]);
                else if (q_0 < c.nq && q_1 < c.nq && (!proved_zero[q_0] || !proved_zero[q_1])) {
                    proved_zero[q_0] = proved_zero[q_1] = 0;
                }
            } else if (b == T_CSWAP) {
                if (q_0 < c.nq && q_1 < c.nq && q_2 < c.nq && !proved_zero[q_0] && (!proved_zero[q_1] || !proved_zero[q_2])) {
                    proved_zero[q_1] = proved_zero[q_2] = 0;
                }
            }

            u32 lv = count_nonzero();
            if (lv > peak_live) {
                peak_live = lv;
                first_peak_op = last_peak_op = k;
            } else if (lv == peak_live) {
                last_peak_op = k;
            }
        }

        // Sample 256-shot semantic GF(2) affine rank & affine-dependent wires at first_peak_op + 1
        u64 probe_cut = std::min<u64>(c.num_ops, first_peak_op + 1);
        SimState256 st;
        st.init(c.nq, c.nb, seed);
        init_random_inputs(st, c.nq, clean, 256);
        if (probe_cut > 0) run_span_256(c, 0, probe_cut, st, 0, 256);

        // Build GF(2) basis over 256-bit rows (including constant-1 row for affine combinations)
        struct BasisRow {
            Word256 vec;
            std::vector<int> combo;  // -1 denotes constant 1, >=0 denotes wire id
        };
        std::map<int, BasisRow> pivots;
        auto first_bit = [](const Word256& v) -> int {
            for (int w = 0; w < 4; ++w) {
                if (v.w[w]) return w * 64 + __builtin_ctzll(v.w[w]);
            }
            return -1;
        };
        auto xor_combo = [](std::vector<int>& a, const std::vector<int>& b) {
            std::vector<int> out;
            size_t i = 0, j = 0;
            while (i < a.size() && j < b.size()) {
                if (a[i] < b[j]) out.push_back(a[i++]);
                else if (b[j] < a[i]) out.push_back(b[j++]);
                else { ++i; ++j; }
            }
            while (i < a.size()) out.push_back(a[i++]);
            while (j < b.size()) out.push_back(b[j++]);
            a.swap(out);
        };

        // Insert constant-1 vector first
        {
            Word256 ones;
            ones.w[0] = ones.w[1] = ones.w[2] = ones.w[3] = ~0ull;
            pivots[0] = BasisRow{ones, {-1}};
        }

        PyObject* loanable = PyList_New(0);
        u32 nonzero_sampled = 0;
        for (u32 q = 0; q < c.nq; ++q) {
            Word256 v = st.q[q];
            if ((v.w[0] | v.w[1] | v.w[2] | v.w[3]) == 0) continue;
            nonzero_sampled++;
            std::vector<int> comb = {static_cast<int>(q)};
            while (true) {
                int p = first_bit(v);
                if (p < 0) break;
                auto it = pivots.find(p);
                if (it == pivots.end()) {
                    pivots[p] = BasisRow{v, std::move(comb)};
                    comb.clear();
                    break;
                }
                for (int w = 0; w < 4; ++w) v.w[w] ^= it->second.vec.w[w];
                xor_combo(comb, it->second.combo);
            }
            if (!comb.empty()) {
                // Wire q equals XOR of peers in comb (excluding q itself)!
                bool const_one = false;
                PyObject* peers = PyList_New(0);
                for (int id : comb) {
                    if (id == static_cast<int>(q)) continue;
                    if (id == -1) const_one = true;
                    else {
                        PyObject* py_id = PyLong_FromLong(id);
                        PyList_Append(peers, py_id);
                        Py_DECREF(py_id);
                    }
                }
                PyObject* item = PyDict_New();
                PyDict_SetItemString(item, "wire", PyLong_FromUnsignedLong(q));
                PyDict_SetItemString(item, "constant", PyLong_FromLong(const_one ? 1 : 0));
                PyDict_SetItemString(item, "xor_peers", peers);
                Py_DECREF(peers);
                PyList_Append(loanable, item);
                Py_DECREF(item);
            }
        }
        u32 semantic_affine_rank = static_cast<u32>(pivots.size() - 1);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        PyObject* res = PyDict_New();
        PyDict_SetItemString(res, "num_qubits", PyLong_FromUnsignedLong(c.nq));
        PyDict_SetItemString(res, "num_ops", PyLong_FromUnsignedLongLong(c.num_ops));
        PyDict_SetItemString(res, "r55_peak_live_qubits", PyLong_FromUnsignedLong(peak_live));
        PyDict_SetItemString(res, "first_peak_op", PyLong_FromUnsignedLongLong(first_peak_op));
        PyDict_SetItemString(res, "last_peak_op", PyLong_FromUnsignedLongLong(last_peak_op));
        PyDict_SetItemString(res, "sampled_active_at_peak", PyLong_FromUnsignedLong(nonzero_sampled));
        PyDict_SetItemString(res, "semantic_affine_rank_at_peak", PyLong_FromUnsignedLong(semantic_affine_rank));
        PyDict_SetItemString(res, "affine_loan_candidates", loanable); Py_DECREF(loanable);
        PyDict_SetItemString(res, "classical_prefix_ops", PyLong_FromUnsignedLongLong(classical_prefix_ops));
        PyDict_SetItemString(res, "hmr_sites", PyLong_FromUnsignedLongLong(hmr_count));
        PyDict_SetItemString(res, "elapsed_ms", PyFloat_FromDouble(ms));
        return res;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

static PyMethodDef Engine_methods[] = {
    {"trace_cuts", reinterpret_cast<PyCFunction>(Engine_trace_cuts), METH_VARARGS | METH_KEYWORDS, "Bit-sliced multi-cut packed trace & phase extractor."},
    {"verify_cone", reinterpret_cast<PyCFunction>(Engine_verify_cone), METH_VARARGS | METH_KEYWORDS, "Checkpointed causal-cone differential replay."},
    {"verify_cones_batch", reinterpret_cast<PyCFunction>(Engine_verify_cones_batch), METH_VARARGS | METH_KEYWORDS, "OpenMP batched candidate cone evaluator."},
    {"profile_liveness", reinterpret_cast<PyCFunction>(Engine_profile_liveness), METH_VARARGS | METH_KEYWORDS, "R55 proved-zero, semantic GF(2) rank, and affine wire-loan profiler."},
    {nullptr, nullptr, 0, nullptr}
};

static PyTypeObject BitSlicedEngineType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
};

// ---------------------------------------------------------------------------
// Module-Level Fast Linear Algebra & Matroid-Parity Functions
// ---------------------------------------------------------------------------

// gf2_rank_packed(packed_bytes, num_funcs, row_count) -> (rank, witness_cols)
static PyObject* mod_gf2_rank_packed(PyObject*, PyObject* args) {
    Py_buffer buf{};
    int num_funcs = 0, row_count = 0;
    if (!PyArg_ParseTuple(args, "y*ii", &buf, &num_funcs, &row_count)) return nullptr;
    try {
        size_t row_bytes = static_cast<size_t>((row_count + 7) / 8);
        if (static_cast<size_t>(buf.len) != static_cast<size_t>(num_funcs) * row_bytes) {
            PyBuffer_Release(&buf);
            throw std::runtime_error("Buffer size mismatch in gf2_rank_packed");
        }
        size_t nwords = static_cast<size_t>((row_count + 63) / 64);
        const auto* raw = static_cast<const u8*>(buf.buf);

        struct Piv {
            std::vector<u64> w;
            int col;
        };
        std::map<int, Piv> basis;
        for (int r = 0; r < num_funcs; ++r) {
            std::vector<u64> vec(nwords, 0);
            const u8* src = raw + static_cast<size_t>(r) * row_bytes;
            for (int c = 0; c < row_count; ++c) {
                if ((src[c >> 3] >> (7 - (c & 7))) & 1u) {
                    vec[static_cast<size_t>(c >> 6)] |= (1ull << (c & 63));
                }
            }
            while (true) {
                int p = -1;
                for (size_t wi = 0; wi < nwords; ++wi) {
                    if (vec[wi]) {
                        p = static_cast<int>(wi * 64 + __builtin_ctzll(vec[wi]));
                        break;
                    }
                }
                if (p < 0) break;
                auto it = basis.find(p);
                if (it == basis.end()) {
                    basis[p] = Piv{std::move(vec), p};
                    break;
                }
                for (size_t wi = 0; wi < nwords; ++wi) vec[wi] ^= it->second.w[wi];
            }
        }
        PyBuffer_Release(&buf);

        PyObject* cols = PyList_New(static_cast<Py_ssize_t>(basis.size()));
        Py_ssize_t idx = 0;
        for (const auto& kv : basis) {
            PyList_SET_ITEM(cols, idx++, PyLong_FromLong(kv.first));
        }
        PyObject* res = PyTuple_New(2);
        PyTuple_SET_ITEM(res, 0, PyLong_FromSize_t(basis.size()));
        PyTuple_SET_ITEM(res, 1, cols);
        return res;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

// solve_zmod_linear(features_u8, target_u8, n_samples, n_features, modulus)
// Solves F * x = target (mod modulus) for modulus in {2, 4, 8} where features_u8 has shape (n_samples, n_features).
static PyObject* mod_solve_zmod_linear(PyObject*, PyObject* args) {
    Py_buffer fbuf{}, tbuf{};
    int n_samples = 0, n_features = 0, modulus = 4;
    if (!PyArg_ParseTuple(args, "y*y*iii", &fbuf, &tbuf, &n_samples, &n_features, &modulus)) return nullptr;
    try {
        auto t0 = std::chrono::steady_clock::now();
        if (modulus != 2 && modulus != 4 && modulus != 8) {
            PyBuffer_Release(&fbuf); PyBuffer_Release(&tbuf);
            throw std::runtime_error("modulus must be 2, 4, or 8");
        }
        if (fbuf.len != static_cast<Py_ssize_t>(n_samples) * n_features || tbuf.len != n_samples) {
            PyBuffer_Release(&fbuf); PyBuffer_Release(&tbuf);
            throw std::runtime_error("Shape mismatch in solve_zmod_linear");
        }
        int kbits = (modulus == 2) ? 1 : (modulus == 4 ? 2 : 3);
        const auto* F = static_cast<const u8*>(fbuf.buf);
        const auto* Y = static_cast<const u8*>(tbuf.buf);

        // Expand each integer variable x_j in Z_{2^k} as binary digits x_j = sum_{b=0}^{k-1} 2^b u_{j,b}
        // Carry-free linear system over Z_{2^k} solved via exact 2-adic / Z_{2^k} Gauss-Jordan elimination!
        u8 mask = static_cast<u8>(modulus - 1);
        std::vector<u8> A(static_cast<size_t>(n_samples) * (n_features + 1), 0);
        for (int r = 0; r < n_samples; ++r) {
            for (int c = 0; c < n_features; ++c) {
                A[static_cast<size_t>(r) * (n_features + 1) + c] = F[static_cast<size_t>(r) * n_features + c] & mask;
            }
            A[static_cast<size_t>(r) * (n_features + 1) + n_features] = Y[r] & mask;
        }
        PyBuffer_Release(&fbuf);
        PyBuffer_Release(&tbuf);

        auto inv_odd = [&](u8 v) -> u8 {
            // Odd numbers mod 2, 4, 8 are self-inverse (1*1=1, 3*3=9=1, 5*5=25=1, 7*7=49=1 mod 8)!
            (void)kbits;
            return v & mask;
        };

        int row = 0;
        std::vector<std::pair<int, int>> pivot_rc;
        // Pass 1: unit (odd) pivots; Pass 2: 2*odd pivots; Pass 3: 4*odd pivots
        for (int val_shift = 0; val_shift < kbits && row < n_samples; ++val_shift) {
            u8 pbit = static_cast<u8>(1u << val_shift);
            for (int col = 0; col < n_features && row < n_samples; ++col) {
                int sel = -1;
                for (int r = row; r < n_samples; ++r) {
                    u8 v = A[static_cast<size_t>(r) * (n_features + 1) + col] & mask;
                    if (v != 0 && (v & ((pbit << 1) - 1)) == pbit) {
                        sel = r;
                        break;
                    }
                }
                if (sel < 0) continue;
                if (sel != row) {
                    for (int c = 0; c <= n_features; ++c) {
                        std::swap(A[static_cast<size_t>(sel) * (n_features + 1) + c],
                                  A[static_cast<size_t>(row) * (n_features + 1) + c]);
                    }
                }
                u8 pv = A[static_cast<size_t>(row) * (n_features + 1) + col] & mask;
                u8 odd_part = static_cast<u8>(pv >> val_shift) | 1u;
                u8 inv = inv_odd(odd_part);
                if (inv != 1u) {
                    for (int c = col; c <= n_features; ++c) {
                        A[static_cast<size_t>(row) * (n_features + 1) + c] =
                            (A[static_cast<size_t>(row) * (n_features + 1) + c] * inv) & mask;
                    }
                }
                // Normalized pivot is now exactly pbit = 2^val_shift
                #pragma omp parallel for schedule(static) if(n_samples - row > 512)
                for (int r = 0; r < n_samples; ++r) {
                    if (r == row) continue;
                    u8 target_v = A[static_cast<size_t>(r) * (n_features + 1) + col] & mask;
                    if (target_v != 0 && (target_v % pbit) == 0) {
                        u8 factor = static_cast<u8>(target_v / pbit) & mask;
                        for (int c = col; c <= n_features; ++c) {
                            u8 sub = (factor * A[static_cast<size_t>(row) * (n_features + 1) + c]) & mask;
                            A[static_cast<size_t>(r) * (n_features + 1) + c] =
                                (A[static_cast<size_t>(r) * (n_features + 1) + c] + modulus - sub) & mask;
                        }
                    }
                }
                pivot_rc.emplace_back(row, col);
                row++;
            }
        }

        bool feasible = true;
        int contradiction_row = -1;
        for (int r = row; r < n_samples; ++r) {
            bool all_zero = true;
            for (int c = 0; c < n_features; ++c) {
                if (A[static_cast<size_t>(r) * (n_features + 1) + c] != 0) { all_zero = false; break; }
            }
            if (all_zero && A[static_cast<size_t>(r) * (n_features + 1) + n_features] != 0) {
                feasible = false;
                contradiction_row = r;
                break;
            }
        }

        std::vector<u8> sol(static_cast<size_t>(n_features), 0);
        if (feasible) {
            for (int i = static_cast<int>(pivot_rc.size()) - 1; i >= 0; --i) {
                int r = pivot_rc[static_cast<size_t>(i)].first;
                int col = pivot_rc[static_cast<size_t>(i)].second;
                u8 rhs = A[static_cast<size_t>(r) * (n_features + 1) + n_features] & mask;
                for (int c = 0; c < n_features; ++c) {
                    if (c == col) continue;
                    rhs = (rhs + modulus - ((A[static_cast<size_t>(r) * (n_features + 1) + c] * sol[static_cast<size_t>(c)]) & mask)) & mask;
                }
                u8 pv = A[static_cast<size_t>(r) * (n_features + 1) + col] & mask;
                if (rhs == 0) {
                    sol[static_cast<size_t>(col)] = 0;
                } else if (pv != 0 && (rhs % pv) == 0) {
                    sol[static_cast<size_t>(col)] = static_cast<u8>(rhs / pv) & mask;
                } else {
                    feasible = false;
                    contradiction_row = r;
                    break;
                }
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        PyObject* out = PyDict_New();
        PyDict_SetItemString(out, "fits", PyBool_FromLong(feasible ? 1 : 0));
        PyDict_SetItemString(out, "modulus", PyLong_FromLong(modulus));
        PyDict_SetItemString(out, "rank", PyLong_FromSize_t(pivot_rc.size()));
        PyDict_SetItemString(out, "elapsed_ms", PyFloat_FromDouble(ms));
        if (feasible) {
            PyObject* coeffs = PyList_New(n_features);
            int nnz = 0;
            for (int j = 0; j < n_features; ++j) {
                if (sol[static_cast<size_t>(j)]) nnz++;
                PyList_SET_ITEM(coeffs, j, PyLong_FromUnsignedLong(sol[static_cast<size_t>(j)]));
            }
            PyDict_SetItemString(out, "coefficients", coeffs); Py_DECREF(coeffs);
            PyDict_SetItemString(out, "nonzero_count", PyLong_FromLong(nnz));
        } else {
            PyDict_SetItemString(out, "contradiction_row", PyLong_FromLong(contradiction_row));
        }
        return out;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

// lovasz_matroid_parity_rank(lhs_u64, rhs_u64, dim, trials=3, seed=1)
// Computes exact linear matroid parity number max |P| (maximum number of disjoint 2-flats)
// via Lovasz's skew-symmetric wedge matrix M = sum_i x_i (u_i ^ v_i) over GF(2^64).
static PyObject* mod_lovasz_matroid_parity_rank(PyObject*, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"lhs", "rhs", "dim", "trials", "seed", nullptr};
    Py_buffer lbuf{}, rbuf{};
    int dim = 0, trials = 3;
    unsigned long long seed = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "y*y*i|iK", const_cast<char**>(kwlist), &lbuf, &rbuf, &dim, &trials, &seed)) {
        return nullptr;
    }
    try {
        auto t0 = std::chrono::steady_clock::now();
        if (dim < 1 || dim > 64) {
            PyBuffer_Release(&lbuf); PyBuffer_Release(&rbuf);
            throw std::runtime_error("dim must be in 1..64 for packed u64 vectors");
        }
        if (lbuf.len != rbuf.len || (lbuf.len % 8) != 0) {
            PyBuffer_Release(&lbuf); PyBuffer_Release(&rbuf);
            throw std::runtime_error("lhs and rhs must be equal-length uint64 arrays");
        }
        size_t m = static_cast<size_t>(lbuf.len / 8);
        const auto* lhs = static_cast<const u64*>(lbuf.buf);
        const auto* rhs = static_cast<const u64*>(rbuf.buf);

        SimState256 rng_helper;
        rng_helper.init(0, 0, seed ? seed : 1ull);
        int best_matching = 0;

        for (int tr = 0; tr < std::max(1, trials); ++tr) {
            std::vector<u64> M(static_cast<size_t>(dim) * dim, 0);
            for (size_t k = 0; k < m; ++k) {
                u64 u = lhs[k], v = rhs[k];
                if (u == 0 || v == 0 || u == v) continue;
                u64 x = rng_helper.next_rnd();
                if (x == 0) x = 1ull;
                for (int i = 0; i < dim; ++i) {
                    u64 ui = (u >> i) & 1ull;
                    u64 vi = (v >> i) & 1ull;
                    if (!ui && !vi) continue;
                    for (int j = i + 1; j < dim; ++j) {
                        u64 uj = (u >> j) & 1ull;
                        u64 vj = (v >> j) & 1ull;
                        if ((ui & vj) ^ (vi & uj)) {
                            M[static_cast<size_t>(i) * dim + j] ^= x;
                            M[static_cast<size_t>(j) * dim + i] ^= x;
                        }
                    }
                }
            }
            // Gaussian elimination over GF(2^64)
            int r = 0;
            for (int col = 0; col < dim && r < dim; ++col) {
                int sel = -1;
                for (int i = r; i < dim; ++i) {
                    if (M[static_cast<size_t>(i) * dim + col] != 0) { sel = i; break; }
                }
                if (sel < 0) continue;
                if (sel != r) {
                    for (int j = col; j < dim; ++j) {
                        std::swap(M[static_cast<size_t>(sel) * dim + j], M[static_cast<size_t>(r) * dim + j]);
                    }
                }
                u64 inv_p = gf2_64_inv(M[static_cast<size_t>(r) * dim + col]);
                for (int j = col; j < dim; ++j) {
                    M[static_cast<size_t>(r) * dim + j] = gf2_64_mul(M[static_cast<size_t>(r) * dim + j], inv_p);
                }
                for (int i = r + 1; i < dim; ++i) {
                    u64 f = M[static_cast<size_t>(i) * dim + col];
                    if (!f) continue;
                    for (int j = col; j < dim; ++j) {
                        M[static_cast<size_t>(i) * dim + j] ^= gf2_64_mul(f, M[static_cast<size_t>(r) * dim + j]);
                    }
                }
                r++;
            }
            best_matching = std::max(best_matching, r / 2);
        }
        PyBuffer_Release(&lbuf);
        PyBuffer_Release(&rbuf);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        PyObject* out = PyDict_New();
        PyDict_SetItemString(out, "max_paired_lines", PyLong_FromLong(best_matching));
        PyDict_SetItemString(out, "matrix_rank", PyLong_FromLong(best_matching * 2));
        PyDict_SetItemString(out, "num_pairs", PyLong_FromSize_t(m));
        PyDict_SetItemString(out, "dim", PyLong_FromLong(dim));
        PyDict_SetItemString(out, "elapsed_ms", PyFloat_FromDouble(ms));
        return out;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

// eval_phase_margin(target_turns, feature_masks_u64, angles_turns, tolerances_turns)
// Vectorized OpenMP evaluator of worst wrapped phase error / tolerance over millions of inputs.
static PyObject* mod_eval_phase_margin(PyObject*, PyObject* args) {
    Py_buffer tbuf{}, fbuf{}, abuf{}, tolbuf{};
    if (!PyArg_ParseTuple(args, "y*y*y*y*", &tbuf, &fbuf, &abuf, &tolbuf)) return nullptr;
    try {
        auto t0 = std::chrono::steady_clock::now();
        size_t n = static_cast<size_t>(tbuf.len / sizeof(double));
        size_t m = static_cast<size_t>(abuf.len / sizeof(double));
        if (m > 64 || fbuf.len != static_cast<Py_ssize_t>(n * sizeof(u64)) ||
            tolbuf.len != static_cast<Py_ssize_t>(n * sizeof(double))) {
            PyBuffer_Release(&tbuf); PyBuffer_Release(&fbuf); PyBuffer_Release(&abuf); PyBuffer_Release(&tolbuf);
            throw std::runtime_error("Buffer shape mismatch in eval_phase_margin (max 64 features)");
        }
        const auto* target = static_cast<const double*>(tbuf.buf);
        const auto* masks = static_cast<const u64*>(fbuf.buf);
        const auto* angles = static_cast<const double*>(abuf.buf);
        const auto* tols = static_cast<const double*>(tolbuf.buf);

        double global_worst = -1.0;
        size_t global_idx = 0;

        #pragma omp parallel
        {
            double local_worst = -1.0;
            size_t local_idx = 0;
            #pragma omp for schedule(static) nowait
            for (size_t i = 0; i < n; ++i) {
                u64 mk = masks[i];
                double acc = 0.0;
                while (mk) {
                    int b = __builtin_ctzll(mk);
                    mk &= (mk - 1);
                    if (static_cast<size_t>(b) < m) acc += angles[b];
                }
                double diff = acc - target[i];
                diff -= std::round(diff);  // wrap to [-0.5, 0.5] turns
                double ratio = std::abs(diff) / tols[i];
                if (ratio > local_worst) {
                    local_worst = ratio;
                    local_idx = i;
                }
            }
            #pragma omp critical
            {
                if (local_worst > global_worst) {
                    global_worst = local_worst;
                    global_idx = local_idx;
                }
            }
        }

        PyBuffer_Release(&tbuf); PyBuffer_Release(&fbuf); PyBuffer_Release(&abuf); PyBuffer_Release(&tolbuf);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        PyObject* out = PyDict_New();
        PyDict_SetItemString(out, "worst_error_over_tolerance", PyFloat_FromDouble(global_worst));
        PyDict_SetItemString(out, "worst_index", PyLong_FromSize_t(global_idx));
        PyDict_SetItemString(out, "passed", PyBool_FromLong(global_worst <= 1.0 ? 1 : 0));
        PyDict_SetItemString(out, "elapsed_ms", PyFloat_FromDouble(ms));
        return out;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

static PyMethodDef module_methods[] = {
    {"gf2_rank_packed", mod_gf2_rank_packed, METH_VARARGS, "Word-packed GF(2) rank and witness column finder."},
    {"solve_zmod_linear", mod_solve_zmod_linear, METH_VARARGS, "OpenMP linear solver over Z_2, Z_4, or Z_8."},
    {"lovasz_matroid_parity_rank", reinterpret_cast<PyCFunction>(mod_lovasz_matroid_parity_rank), METH_VARARGS | METH_KEYWORDS, "Randomized GF(2^64) Lovasz linear matroid-parity oracle."},
    {"eval_phase_margin", mod_eval_phase_margin, METH_VARARGS, "OpenMP vectorized wrapped phase error margin evaluator."},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Strata C++20 zero-copy .kmb and bit-sliced SIMD latency engine.",
    -1,
    module_methods,
    nullptr, nullptr, nullptr, nullptr
};

}  // namespace

PyMODINIT_FUNC PyInit__native(void) {
    KmbIndexType.tp_name = "strata._native.KmbIndex";
    KmbIndexType.tp_basicsize = sizeof(KmbIndexObject);
    KmbIndexType.tp_itemsize = 0;
    KmbIndexType.tp_flags = Py_TPFLAGS_DEFAULT;
    KmbIndexType.tp_doc = "Zero-copy mmap .kmb / .kmx columnar circuit index.";
    KmbIndexType.tp_new = KmbIndex_new;
    KmbIndexType.tp_init = reinterpret_cast<initproc>(KmbIndex_init);
    KmbIndexType.tp_dealloc = reinterpret_cast<destructor>(KmbIndex_dealloc);
    KmbIndexType.tp_methods = KmbIndex_methods;
    KmbIndexType.tp_as_sequence = &KmbIndex_seq;
    if (PyType_Ready(&KmbIndexType) < 0) return nullptr;

    BitSlicedEngineType.tp_name = "strata._native.BitSlicedEngine";
    BitSlicedEngineType.tp_basicsize = sizeof(BitSlicedEngineObject);
    BitSlicedEngineType.tp_itemsize = 0;
    BitSlicedEngineType.tp_flags = Py_TPFLAGS_DEFAULT;
    BitSlicedEngineType.tp_doc = "256-lane bit-sliced simulator with checkpointed causal-cone replay.";
    BitSlicedEngineType.tp_new = Engine_new;
    BitSlicedEngineType.tp_init = reinterpret_cast<initproc>(Engine_init);
    BitSlicedEngineType.tp_dealloc = reinterpret_cast<destructor>(Engine_dealloc);
    BitSlicedEngineType.tp_methods = Engine_methods;
    if (PyType_Ready(&BitSlicedEngineType) < 0) return nullptr;

    PyObject* m = PyModule_Create(&moduledef);
    if (!m) return nullptr;

    Py_INCREF(&KmbIndexType);
    if (PyModule_AddObject(m, "KmbIndex", reinterpret_cast<PyObject*>(&KmbIndexType)) < 0) {
        Py_DECREF(&KmbIndexType);
        Py_DECREF(m);
        return nullptr;
    }

    Py_INCREF(&BitSlicedEngineType);
    if (PyModule_AddObject(m, "BitSlicedEngine", reinterpret_cast<PyObject*>(&BitSlicedEngineType)) < 0) {
        Py_DECREF(&BitSlicedEngineType);
        Py_DECREF(m);
        return nullptr;
    }

#ifdef _OPENMP
    PyModule_AddIntConstant(m, "OPENMP_ENABLED", 1);
    PyModule_AddIntConstant(m, "MAX_THREADS", omp_get_max_threads());
#else
    PyModule_AddIntConstant(m, "OPENMP_ENABLED", 0);
    PyModule_AddIntConstant(m, "MAX_THREADS", 1);
#endif
    return m;
}
