import argparse as args

class Parameter():
    
    def __init__(self):
        parser = args.ArgumentParser()

        parser.add_argument('--train', dest='train', action='store_true')
        parser.add_argument('--no-train', dest='train', action='store_false')
        parser.set_defaults(train=True)

        parser.add_argument('--convert', dest='convert', action='store_true')
        parser.set_defaults(convert=False)

        #dataloader
        parser.add_argument('--data_path', default='./data/')
        parser.add_argument('--save_path', default='./data/saved')
        parser.add_argument('--img_size', default=160)
        parser.add_argument('--val_percentage', default=5)

        #training
        parser.add_argument('--batch_size', default=32)
        parser.add_argument('--epochs', default=80)
        parser.add_argument('--lr', default=0.0001)
        parser.add_argument('--shuffle_buffer_size', default=6400)
        parser.add_argument('--model_path', default='models/model-v5.h5')
        parser.add_argument('--log_path', default='logs/log.txt')

        #model
        parser.add_argument('--untrainable_layer', default=100)

        self.parsed = parser.parse_args()

    def get_args(self):
        return self.parsed