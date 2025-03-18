from utils.data_execution import load_data


import argparse

def main(args):
    data = load_data(args.data_file)

    ####### BENCHMARK #######


    if args.benchmark_name == "Vispeak":
        from benchmark.vispeak_bench import VispeakBench
        benchmark = VispeakBench(args.data_file, args.video_root)
    if args.benchmark_name == "VispeakProactive":
        from benchmark.vispeak_bench_proactive import VispeakBenchProactive
        benchmark = VispeakBenchProactive(args.data_file, args.video_root)
  

    ##########################

    
    if "VITA1P5" in args.model_name:
        from model.VITA1P5 import VITA
        model = VITA()
    if "OLA" in args.model_name:
        from model.Ola import OLA
        model = OLA()
    if "qwen" in args.model_name:
        from model.Qwenvl import Qwen2P5VL
        model = Qwen2P5VL()
    if "intern" in args.model_name:
        from model.InternVL import InternVL
        model = InternVL()
    if "flashv" in args.model_name:
        from model.FlashVstream import FlashVstream
        model = FlashVstream()
    
    if "dispider" in args.model_name:
        from model.Dispider import Dispider
        model = Dispider()


    ######################

    benchmark.eval(data, model, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, required=True, help="Path to the data file")
    parser.add_argument("--video-root", type=str, required=False, help="Path to the video dictionary", default="./data")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model")
    parser.add_argument("--benchmark-name", type=str, required=True, help="Name of the benchmark")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output file")
    args = parser.parse_args()
    main(args)