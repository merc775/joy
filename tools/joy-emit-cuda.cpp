//===- joy-emit-cuda.cpp - MLIR -> CUDA C source dumper -----------------===//
//
// Tiny driver that:
//   1. parses an MLIR file (post-codegen-rms-norm form),
//   2. optionally runs `--codegen-rms-norm` itself if asked,
//   3. invokes joy::emitCudaC on the result and writes CUDA source to
//      either stdout or a -o file.
//
// Typical pipeline (used by scripts/regen_codegen_kernel.sh):
//
//   joy-opt --lower-joy-to-joyl --codegen-rms-norm stub.mlir
//     | joy-emit-cuda - -o codegen_kernel.cu
//
//===----------------------------------------------------------------------===//

#include "joy/dialect/joy/JoyDialect.h"
#include "joy/dialect/joyh/JoyhDialect.h"
#include "joy/dialect/joyl/JoylDialect.h"
#include "joy/optimizer/EmitCudaC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

void registerCodegenRMSNormPass();
void registerLowerJoyToJoylPass();
void registerOpFusionPass();
void registerLowerJoylToJoyhPass();

namespace {
llvm::cl::opt<std::string> InputFilename(llvm::cl::Positional,
                                          llvm::cl::desc("<input mlir>"),
                                          llvm::cl::init("-"));
llvm::cl::opt<std::string> OutputFilename(
    "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
    llvm::cl::init("-"));
llvm::cl::opt<bool> RunPipeline(
    "run-pipeline",
    llvm::cl::desc("Run --lower-joy-to-joyl --codegen-rms-norm before emit"),
    llvm::cl::init(false));
llvm::cl::opt<std::string> SourceTag(
    "source-tag",
    llvm::cl::desc("String to embed in the // Source: header line"),
    llvm::cl::init(""));
} // namespace

static int run(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  registerCodegenRMSNormPass();
  registerLowerJoyToJoylPass();
  registerOpFusionPass();
  registerLowerJoylToJoyhPass();

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                     "joy MLIR -> CUDA C emitter\n");

  std::string errorMessage;
  auto buf = llvm::MemoryBuffer::getFileOrSTDIN(InputFilename);
  if (!buf) {
    llvm::errs() << "joy-emit-cuda: cannot read " << InputFilename << ": "
                 << buf.getError().message() << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*buf), llvm::SMLoc());

  mlir::DialectRegistry registry;
  registry.insert<mlir::joy::JoyDialect, mlir::joyl::JoylDialect,
                  mlir::joyh::JoyhDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect, mlir::math::MathDialect>();

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context));
  if (!module) {
    llvm::errs() << "joy-emit-cuda: failed to parse " << InputFilename << "\n";
    return 1;
  }

  if (RunPipeline) {
    mlir::PassManager pm(&context);
    if (failed(mlir::parsePassPipeline(
            "builtin.module(lower-joy-to-joyl,codegen-rms-norm)", pm))) {
      llvm::errs() << "joy-emit-cuda: failed to parse internal pass pipeline\n";
      return 1;
    }
    if (failed(pm.run(*module))) {
      llvm::errs()
          << "joy-emit-cuda: pass pipeline failed before emission\n";
      return 1;
    }
  }

  std::error_code ec;
  std::unique_ptr<llvm::ToolOutputFile> out;
  if (OutputFilename == "-") {
    if (mlir::failed(joy::emitCudaC(*module, llvm::outs(),
                                     SourceTag.empty()
                                         ? ""
                                         : "// Source: " + SourceTag))) {
      llvm::errs() << "joy-emit-cuda: emitCudaC failed\n";
      return 1;
    }
    return 0;
  }
  out.reset(new llvm::ToolOutputFile(OutputFilename, ec,
                                      llvm::sys::fs::OF_None));
  if (ec) {
    llvm::errs() << "joy-emit-cuda: cannot open " << OutputFilename << ": "
                 << ec.message() << "\n";
    return 1;
  }
  if (mlir::failed(joy::emitCudaC(*module, out->os(),
                                    SourceTag.empty()
                                        ? ""
                                        : "// Source: " + SourceTag))) {
    llvm::errs() << "joy-emit-cuda: emitCudaC failed\n";
    return 1;
  }
  out->keep();
  return 0;
}

int main(int argc, char **argv) { return run(argc, argv); }
