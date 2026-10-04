#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <array>
#include <algorithm>
using namespace std;

static constexpr uint32_t NNUE_VERSION = 0x7AF32F16;
static constexpr uint32_t MAIN_HASH = 0xc203b032;
static constexpr uint32_t FT_HASH = 0x5f134ab8;
static constexpr size_t INPUT_SIZE=131949, FT_SIZE=1024, L1_OUT=8, L2_IN=14, L2_PAD=32, L2_SIZE=64, L3_IN=64, STACKS=72;
static constexpr float QA=127.0f, QB=64.0f, BS=8128.0f;
static const string ARCH="ModelType=SFNNWithoutPsqt;Features=HalfKA2(Friend)[131949->1024x2],Network=SFNN_HALFKA2_1024_7_64_K3K3_PROGRESS8{LayerStack=72}";
static const char MAGIC[]="COMPRESSED_LEB128";

uint64_t read_u64(ifstream&f){uint64_t v;f.read((char*)&v,8);if(!f)throw runtime_error("u64");return v;}
void write_u32(ofstream&o,uint32_t v){o.write((char*)&v,4);}
void write_i32(ofstream&o,int32_t v){o.write((char*)&v,4);}
int q16(float x,float s){ long v=lround((double)x*s); return (int)max<long>(-32768,min<long>(32767,v)); }
int q8(float x){ long v=lround((double)x*QB); return (int)max<long>(-128,min<long>(127,v)); }
int32_t q32(float x){ long long v=llround((double)x*BS); if(v<INT32_MIN)v=INT32_MIN; if(v>INT32_MAX)v=INT32_MAX; return (int32_t)v; }

void sleb16(vector<uint8_t>&b,int16_t val){
    int32_t v=val; bool more=true;
    while(more){
        uint8_t byte=v&0x7f; bool sign=byte&0x40; v >>= 7;
        if((v==0 && !sign)||(v==-1 && sign)) more=false; else byte|=0x80;
        b.push_back(byte);
    }
}
void write_leb(ofstream&o,const vector<float>&v,float scale){
    vector<uint8_t>b; b.reserve(v.size()*2);
    for(float x:v) sleb16(b,(int16_t)q16(x,scale));
    o.write(MAGIC,strlen(MAGIC)); uint32_t n=(uint32_t)b.size(); write_u32(o,n); o.write((char*)b.data(),b.size());
}
map<string,vector<float>> read_state(const filesystem::path&p){
    ifstream f(p,ios::binary); if(!f) throw runtime_error("cannot open state.bin");
    map<string,vector<float>> m;
    while(true){
        string id; char c;
        while(f.get(c)){ if(c=='\n') break; id.push_back(c); }
        if(!f){ if(id.empty()) break; throw runtime_error("truncated id");}
        uint64_t n=read_u64(f); vector<float> v(n); f.read((char*)v.data(),n*4); if(!f)throw runtime_error("truncated "+id);
        m[id]=move(v);
    }
    return m;
}
const vector<float>& req(const map<string,vector<float>>&m,const string&k,size_t n){
    auto it=m.find(k); if(it==m.end()) throw runtime_error("missing "+k);
    if(it->second.size()!=n) throw runtime_error("bad length "+k+" got "+to_string(it->second.size())+" expected "+to_string(n));
    return it->second;
}
int main(int argc,char**argv){
 try{
  if(argc!=3){cerr<<"usage: bulletou_state_to_nn state.bin roundtrip_nn.bin\n";return 2;}
  auto m=read_state(argv[1]);
  auto& l0w=req(m,"nnue/weights/l0w",INPUT_SIZE*FT_SIZE);
  auto& l0b=req(m,"nnue/weights/l0b",FT_SIZE);
  auto& l1w=req(m,"nnue/weights/l1w",STACKS*L1_OUT*FT_SIZE);
  auto& l1b=req(m,"nnue/weights/l1b",STACKS*L1_OUT);
  auto& l2w=req(m,"nnue/weights/l2w",STACKS*L2_SIZE*L2_IN);
  auto& l2b=req(m,"nnue/weights/l2b",STACKS*L2_SIZE);
  auto& l3w=req(m,"nnue/weights/l3w",STACKS*L3_IN);
  auto& l3b=req(m,"nnue/weights/l3b",STACKS);
  ofstream o(argv[2],ios::binary); if(!o)throw runtime_error("cannot create output");
  write_u32(o,NNUE_VERSION); write_u32(o,MAIN_HASH); write_u32(o,(uint32_t)ARCH.size()); o.write(ARCH.data(),ARCH.size()); write_u32(o,FT_HASH);
  write_leb(o,l0b,QA); write_leb(o,l0w,QA);
  for(size_t st=0;st<STACKS;st++){
    write_u32(o,0x3e5aa6ee);
    for(size_t i=0;i<L1_OUT;i++) write_i32(o,q32(l1b[st*L1_OUT+i]));
    for(size_t i=0;i<L1_OUT*FT_SIZE;i++){ int8_t q=(int8_t)q8(l1w[st*L1_OUT*FT_SIZE+i]); o.write((char*)&q,1);}
    for(size_t i=0;i<L2_SIZE;i++) write_i32(o,q32(l2b[st*L2_SIZE+i]));
    for(size_t r=0;r<L2_SIZE;r++) for(size_t c=0;c<L2_PAD;c++){
      int8_t q=0; if(c<L2_IN) q=(int8_t)q8(l2w[st*L2_SIZE*L2_IN+r*L2_IN+c]); o.write((char*)&q,1);
    }
    write_i32(o,q32(l3b[st]));
    for(size_t c=0;c<L3_IN;c++){int8_t q=(int8_t)q8(l3w[st*L3_IN+c]);o.write((char*)&q,1);}
  }
  o.flush(); if(!o)throw runtime_error("write failed");
  cout<<"PASS: state.bin structure accepted and nn.bin re-exported.\n";
  cout<<"output="<<filesystem::absolute(argv[2]).string()<<" bytes="<<filesystem::file_size(argv[2])<<"\n";
  return 0;
 }catch(const exception&e){cerr<<"ERROR: "<<e.what()<<"\n";return 1;}
}
