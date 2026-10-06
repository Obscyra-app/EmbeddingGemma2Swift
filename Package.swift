// swift-tools-version: 6.3
import PackageDescription
let package = Package(
    name: "EmbeddingGemma2Swift",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "EmbeddingGemma2", targets: ["EmbeddingGemma2"]),
        .executable(name: "eg2-embed", targets: ["EmbeddingGemma2CLI"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.32.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.0")
    ],
    targets: [
        .target(name: "EmbeddingGemma2", dependencies: [
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "Tokenizers", package: "swift-transformers")]),
        .executableTarget(name: "EmbeddingGemma2CLI", dependencies: ["EmbeddingGemma2"]),
        .testTarget(name: "EmbeddingGemma2Tests", dependencies: ["EmbeddingGemma2"])
    ])
