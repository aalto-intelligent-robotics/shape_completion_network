from shape_reconstruction.utils import argparser
import shape_completion
if __name__ == "__main__":
    parser = argparser.train_test_parser()
    args = parser.parse_args()

    if args.mode == "train":
        print("Training")
        shape_completion.train_model(args)
    elif args.mode == "test":
        print("Testing")
        shape_completion.test_model(args)
