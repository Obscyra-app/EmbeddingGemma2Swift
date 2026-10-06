import SwiftUI
import EmbeddingGemma2
import CryptoKit
import MLX

struct Reference: Decodable {
    let input_ids: [[Int]]
    let attention_mask: [[Int]]
    let vectors: [String: [[Float]]]
}
struct Inputs: Decodable { let texts: [String] }
struct VisionReference: Decodable {
    let max_soft_tokens, soft_tokens: Int
    let resized_size: [Int]
    let resized_rgb_sha256: String
    let input_ids: [[Int]]
    let vectors: [String:[[Float]]]
}
struct AudioReference: Decodable {
    let sample_count, frames, valid_frames, soft_tokens: Int
    let input_ids: [[Int]]
    let vectors: [String:[[Float]]]
}
@main struct DemoApp: App {
    var body: some Scene { WindowGroup { ValidationView() } }
}
struct ValidationView: View {
    @State private var status = "Ready"
    @State private var running = false
    @State private var weightFormat = ""
    var body: some View {
        VStack(alignment:.leading,spacing:20) {
            Text("EmbeddingGemma 2").font(.largeTitle)
            Text("Native MLX Swift · Text + Images + Audio · \(weightFormat)").foregroundStyle(.secondary)
            ScrollView { Text(status).font(.system(.body,design:.monospaced)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading) }
            Button("Run validation") { Task { await validate() } }.disabled(running)
        }.padding().task { await validate() }
    }
    @MainActor private func validate() async {
        guard !running else { return };running=true;defer { running=false }
        status="Loading local model…"
        do {
            let root=Bundle.main.resourceURL!
            // Application policy: keep reusable GPU cache modest on the phone.
            Memory.cacheLimit = 64*1024*1024
            let start=Date()
            let encoder=try await TextEncoder(directory:root.appendingPathComponent("Model"))
            weightFormat=await encoder.weightFormat()
            let loadTime=Date().timeIntervalSince(start)
            var checks=[[String:Any]]()
            for name in ["multilingual","long"] {
                status="Checking \(name)…"
                let refs=try JSONDecoder().decode(Reference.self,from:Data(contentsOf:root.appendingPathComponent("Fixtures/\(name)-reference.json")))
                let request=try JSONDecoder().decode(Inputs.self,from:Data(contentsOf:root.appendingPathComponent("Fixtures/\(name)-request.json")))
                let ids=try await encoder.tokenize(request.texts)
                let expectedIDs=zip(refs.input_ids,refs.attention_mask).map { row,mask in zip(row,mask).filter { $0.1 == 1 }.map { $0.0 } }
                let t=Date();let vectors=try await encoder.encode(request.texts)
                let seconds=Date().timeIntervalSince(t)
                var minimumCosine=Float(1),maximumError=Float(0),maximumNormError=Float(0)
                for (v,r) in zip(vectors,refs.vectors["768"]!) {
                    let dot=zip(v,r).reduce(Float(0)) { $0+$1.0*$1.1 }
                    let nv=sqrt(v.reduce(Float(0)) { $0+$1*$1 }),nr=sqrt(r.reduce(Float(0)) { $0+$1*$1 })
                    minimumCosine=min(minimumCosine,dot/(nv*nr));maximumNormError=max(maximumNormError,abs(nv-1))
                    maximumError=max(maximumError,zip(v,r).map { abs($0-$1) }.max()!)
                }
                let passed=minimumCosine>=0.999 && maximumError<=0.01 && maximumNormError<=0.00001 && ids==expectedIDs && vectors.flatMap{$0}.allSatisfy(\.isFinite)
                checks.append(["case":name,"pass":passed,"minimum_cosine":minimumCosine,"maximum_absolute_error":maximumError,"maximum_norm_error":maximumNormError,"tokenization_match":ids==expectedIDs,"seconds":seconds,"embeddings":vectors])
            }
            status="Loading vision weights…"
            let visionStart=Date()
            try await encoder.enableVision(directory:root.appendingPathComponent("Vision"))
            let visionLoadTime=Date().timeIntervalSince(visionStart)
            var visionChecks=[[String:Any]]()
            for name in ["landscape70","portrait140","square280","downscale70"] {
                status="Checking image: \(name)…"
                let reference=try JSONDecoder().decode(VisionReference.self,from:Data(contentsOf:root.appendingPathComponent("Fixtures/\(name)-reference.json")))
                let t=Date()
                let prepared=try ImageProcessor.prepare(RGBImage.load(root.appendingPathComponent("Fixtures/\(name).png")),maxSoftTokens:reference.max_soft_tokens)
                let hash=SHA256.hash(data:Data(prepared.resizedRGB)).map { String(format:"%02x",$0) }.joined()
                let ids=try await encoder.imageTokenIDs(softTokenCount:prepared.softTokenCount)
                let v=try await encoder.encodeImage(prepared)
                let seconds=Date().timeIntervalSince(t)
                let r=reference.vectors["768"]![0]
                let dot=zip(v,r).reduce(Float(0)) { $0+$1.0*$1.1 }
                let norm=sqrt(v.reduce(Float(0)) { $0+$1*$1 })
                let refNorm=sqrt(r.reduce(Float(0)) { $0+$1*$1 })
                let cosine=dot/(norm*refNorm),error=zip(v,r).map { abs($0-$1) }.max()!
                let preprocess=hash==reference.resized_rgb_sha256 && [prepared.width,prepared.height]==reference.resized_size && prepared.softTokenCount==reference.soft_tokens
                let passed=v.count==768 && v.allSatisfy(\.isFinite) && cosine>=0.999 && error<=0.01 && abs(norm-1)<=0.00001 && [ids]==reference.input_ids && preprocess
                let metadata:[String:Any]=["resized_size":[prepared.width,prepared.height],"soft_tokens":prepared.softTokenCount,"max_soft_tokens":prepared.maxSoftTokens,"resized_rgb_sha256":hash]
                visionChecks.append(["case":name,"pass":passed,"minimum_cosine":cosine,"maximum_absolute_error":error,"seconds":seconds,"encode_seconds":[seconds],"embeddings":[v],"input_ids":[ids],"images":[metadata],"dimensions":768,"precision":"bfloat16"])
            }
            await encoder.disableVision()
            status="Loading audio weights…"
            let audioStart=Date()
            try await encoder.enableAudio(directory:root.appendingPathComponent("Audio"))
            let audioLoadTime=Date().timeIntervalSince(audioStart)
            var audioChecks=[[String:Any]]()
            for name in ["silence","tone","noise","chirp"] {
                status="Checking audio: \(name)…"
                let reference=try JSONDecoder().decode(AudioReference.self,from:Data(contentsOf:root.appendingPathComponent("Fixtures/\(name)-reference.json")))
                let t=Date()
                let prepared=try AudioProcessor.prepare(samples:AudioProcessor.loadWAV(root.appendingPathComponent("Fixtures/\(name).wav")))
                let ids=try await encoder.audioTokenIDs(softTokenCount:prepared.softTokenCount)
                let v=try await encoder.encodeAudio(prepared)
                let seconds=Date().timeIntervalSince(t),r=reference.vectors["768"]![0]
                let norm=sqrt(v.reduce(Float(0)) { $0+$1*$1 }),refNorm=sqrt(r.reduce(Float(0)) { $0+$1*$1 })
                let cosine=zip(v,r).reduce(Float(0)) { $0+$1.0*$1.1 }/(norm*refNorm)
                let error=zip(v,r).map { abs($0-$1) }.max()!
                let valid=prepared.mask.reduce(0,+)
                let metadata:[String:Any]=["sample_count":prepared.sampleCount,"frames":prepared.frames,"valid_frames":valid,"soft_tokens":prepared.softTokenCount]
                let preprocess=prepared.sampleCount==reference.sample_count && prepared.frames==reference.frames && valid==reference.valid_frames && prepared.softTokenCount==reference.soft_tokens
                let passed=v.count==768 && v.allSatisfy(\.isFinite) && cosine>=0.999 && error<=0.01 && abs(norm-1)<=0.00001 && [ids]==reference.input_ids && preprocess
                audioChecks.append(["case":name,"pass":passed,"minimum_cosine":cosine,"maximum_absolute_error":error,"seconds":seconds,"encode_seconds":[seconds],"embeddings":[v],"input_ids":[ids],"audio":[metadata],"dimensions":768,"precision":"bfloat16"])
            }
            let result:[String:Any]=["platform":"iOS","text_weight_format":weightFormat,"os":ProcessInfo.processInfo.operatingSystemVersionString,"precision":"bfloat16","load_seconds":loadTime,"vision_load_seconds":visionLoadTime,"checks":checks,"vision_checks":visionChecks,"audio_checks":audioChecks,"audio_load_seconds":audioLoadTime,"memory":await encoder.memory(),"mlx_cache_limit_bytes":64*1024*1024]
            let data=try JSONSerialization.data(withJSONObject:result,options:[.sortedKeys])
            let path=FileManager.default.urls(for:.documentDirectory,in:.userDomainMask)[0].appendingPathComponent("validation.json")
            try data.write(to:path,options:.atomic)
            print("EMBEDDINGGEMMA2_RESULT "+String(decoding:data,as:UTF8.self))
            status=(checks+visionChecks+audioChecks).map { "\($0["case"]!): \(($0["pass"] as! Bool) ? "PASS":"FAIL")\ncosine: \($0["minimum_cosine"]!)\ntime: \($0["seconds"]!) s" }.joined(separator:"\n\n")
        } catch {
            status="Error: \(error)";print("EMBEDDINGGEMMA2_ERROR \(error)")
            if let path=FileManager.default.urls(for:.documentDirectory,in:.userDomainMask).first?.appendingPathComponent("error.txt") { try? status.write(to:path,atomically:true,encoding:.utf8) }
        }
    }
}
