#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <filesystem>
#include <array>

using namespace std;

static constexpr const char MAGIC[] = "COMPRESSED_LEB128";
static constexpr size_t MAGIC_LEN = sizeof(MAGIC)-1;
static constexpr uint32_t NNUE_VERSION = 0x7AF32F16;
static constexpr size_t INPUT_SIZE = 131949;
static constexpr size_t FT_SIZE = 1024;
static constexpr size_t L1_OUT = 8;
static constexpr size_t L2_IN = 14;
static constexpr size_t L2_PAD = 32;
static constexpr size_t L2_SIZE = 64;
static constexpr size_t L3_IN = 64;
static constexpr size_t STACKS = 72;
static constexpr float QA = 127.0f;
static constexpr float QB = 64.0f;
static constexpr float BIAS_SCALE = 8128.0f;

uint32_t read_u32(ifstream& f) { uint32_t v; f.read((char*)&v,4); if(!f) throw runtime_error("read u32 failed"); return v; }
int32_t read_i32(ifstream& f) { int32_t v; f.read((char*)&v,4); if(!f) throw runtime_error("read i32 failed"); return v; }
void write_u64(ofstream& o, uint64_t v) { o.write((char*)&v,8); }
void write_record_header(ofstream& o, const string& id, uint64_t n) { o.write(id.data(), id.size()); o.put('\n'); write_u64(o,n); }
void write_f32(ofstream& o, float v) { o.write((char*)&v,4); }
void write_record_vec(ofstream& o, const string& id, const vector<float>& v) { write_record_header(o,id,v.size()); o.write((const char*)v.data(), v.size()*sizeof(float)); }
void write_record_scalar(ofstream& o, const string& id, float v) { write_record_header(o,id,1); write_f32(o,v); }

struct LebBlock { vector<uint8_t> bytes; size_t pos=0; };
LebBlock read_leb_block(ifstream& f) {
    array<char,MAGIC_LEN> m{}; f.read(m.data(), m.size());
    if(!f || memcmp(m.data(),MAGIC,MAGIC_LEN)!=0) throw runtime_error("LEB128 magic mismatch");
    uint32_t n=read_u32(f); LebBlock b; b.bytes.resize(n); f.read((char*)b.bytes.data(),n);
    if(!f) throw runtime_error("LEB128 payload truncated"); return b;
}
int16_t next_sleb16(LebBlock& b) {
    int32_t result=0; int shift=0; uint8_t byte=0;
    do {
        if(b.pos>=b.bytes.size()) throw runtime_error("LEB128 overrun");
        byte=b.bytes[b.pos++]; result |= int32_t(byte & 0x7f) << shift; shift += 7;
    } while((byte & 0x80) && shift < 16);
    if(shift < 16 && (byte & 0x40)) result |= -(1 << shift);
    return (int16_t)result;
}
void write_leb_as_f32_record(ofstream& out, const string& id, LebBlock& b, uint64_t count, float scale) {
    write_record_header(out,id,count);
    vector<float> chunk; chunk.reserve(1<<16);
    for(uint64_t i=0;i<count;i++) {
        int16_t q=next_sleb16(b); float x=float(q)/scale;
        long rq=lround(double(x)*scale);
        if(rq!=q) throw runtime_error("dequant/requant mismatch in "+id);
        chunk.push_back(x);
        if(chunk.size()==chunk.capacity()) { out.write((char*)chunk.data(),chunk.size()*4); chunk.clear(); }
    }
    if(!chunk.empty()) out.write((char*)chunk.data(),chunk.size()*4);
    if(b.pos!=b.bytes.size()) throw runtime_error("LEB128 trailing bytes in "+id);
}

int main(int argc,char**argv){
    try {
        if(argc!=4){ cerr<<"usage: nn_to_bulletou_state <nn.bin> <progress.bin> <state.bin>\n"; return 2; }
        filesystem::path nn=argv[1], prog=argv[2], state=argv[3];
        ifstream f(nn,ios::binary); if(!f) throw runtime_error("cannot open nn.bin");
        uint32_t ver=read_u32(f), hash=read_u32(f), alen=read_u32(f);
        if(ver!=NNUE_VERSION) throw runtime_error("unexpected NNUE version");
        string arch(alen,'\0'); f.read(arch.data(),alen); if(!f) throw runtime_error("header truncated");
        if(arch.find("SFNN_HALFKA2_1024_7_64_K3K3_PROGRESS8")==string::npos || arch.find("LayerStack=72")==string::npos)
            throw runtime_error("this importer only accepts SFNN_halfka2_1024_7_64_k3k3_progress8 (72 stacks)");
        uint32_t fthash=read_u32(f);
        LebBlock l0b=read_leb_block(f), l0w=read_leb_block(f);

        filesystem::create_directories(state.parent_path().empty()?filesystem::path("."):state.parent_path());
        ofstream o(state,ios::binary); if(!o) throw runtime_error("cannot create state.bin");
        write_record_scalar(o,"meta/state_backend/cuda-cpp",1.0f);
        write_record_scalar(o,"nnue/train/completed_steps",0.0f);
        { vector<float> sc={1.0f,0.0f}; write_record_vec(o,"nnue/train/shared_coefficients",sc); }
        write_leb_as_f32_record(o,"nnue/weights/l0b",l0b,FT_SIZE,QA);
        write_leb_as_f32_record(o,"nnue/weights/l0w",l0w,INPUT_SIZE*FT_SIZE,QA);

        vector<float> l1b; l1b.reserve(STACKS*L1_OUT);
        vector<float> l1w; l1w.reserve(STACKS*L1_OUT*FT_SIZE);
        vector<float> l2b; l2b.reserve(STACKS*L2_SIZE);
        vector<float> l2w; l2w.reserve(STACKS*L2_SIZE*L2_IN);
        vector<float> l3b; l3b.reserve(STACKS);
        vector<float> l3w; l3w.reserve(STACKS*L3_IN);
        for(size_t st=0;st<STACKS;st++) {
            (void)read_u32(f);
            for(size_t i=0;i<L1_OUT;i++){ int32_t q=read_i32(f); l1b.push_back(float(q)/BIAS_SCALE); }
            for(size_t i=0;i<L1_OUT*FT_SIZE;i++){ int8_t q; f.read((char*)&q,1); l1w.push_back(float(q)/QB); }
            for(size_t i=0;i<L2_SIZE;i++){ int32_t q=read_i32(f); l2b.push_back(float(q)/BIAS_SCALE); }
            for(size_t r=0;r<L2_SIZE;r++) for(size_t c=0;c<L2_PAD;c++){ int8_t q; f.read((char*)&q,1); if(c<L2_IN) l2w.push_back(float(q)/QB); }
            { int32_t q=read_i32(f); l3b.push_back(float(q)/BIAS_SCALE); }
            for(size_t c=0;c<L3_IN;c++){ int8_t q; f.read((char*)&q,1); l3w.push_back(float(q)/QB); }
            if(!f) throw runtime_error("network payload truncated");
        }
        char extra; if(f.read(&extra,1)) throw runtime_error("trailing bytes after 72 stacks");
        write_record_vec(o,"nnue/weights/l1w",l1w); write_record_vec(o,"nnue/weights/l1b",l1b);
        write_record_vec(o,"nnue/weights/l2w",l2w); write_record_vec(o,"nnue/weights/l2b",l2b);
        write_record_vec(o,"nnue/weights/l3w",l3w); write_record_vec(o,"nnue/weights/l3b",l3b);
        write_record_scalar(o,"nnue/weights/l1_revival_flags",0.0f);
        write_record_scalar(o,"nnue/weights/l2_revival_flags",0.0f);

        ifstream p(prog,ios::binary); if(!p) throw runtime_error("cannot open progress.bin");
        uint64_t psz=filesystem::file_size(prog); if(psz!=81ull*1548ull*8ull) throw runtime_error("unexpected progress.bin size");
        uint64_t pc=psz/8; write_record_header(o,"nnue/weights/progress",pc);
        vector<float> pcbuf; pcbuf.reserve(1<<16);
        for(uint64_t i=0;i<pc;i++){ double d; p.read((char*)&d,8); if(!p) throw runtime_error("progress.bin truncated"); pcbuf.push_back((float)d); if(pcbuf.size()==pcbuf.capacity()){ o.write((char*)pcbuf.data(),pcbuf.size()*4); pcbuf.clear(); }}
        if(!pcbuf.empty()) o.write((char*)pcbuf.data(),pcbuf.size()*4);
        o.flush(); if(!o) throw runtime_error("state.bin write failed");

        cout<<"PASS: BulletOu weights-only initial state created.\n";
        cout<<"arch="<<arch<<"\n";
        cout<<"main_hash=0x"<<hex<<hash<<" ft_hash=0x"<<fthash<<dec<<"\n";
        cout<<"output="<<state.string()<<" bytes="<<filesystem::file_size(state)<<"\n";
        cout<<"Use BulletOu with --initial-state state.bin --sfnn-ft-factorizer false --sfnn-factorized false\n";
        return 0;
    } catch(const exception& e){ cerr<<"ERROR: "<<e.what()<<"\n"; return 1; }
}
