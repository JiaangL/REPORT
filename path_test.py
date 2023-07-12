import torch

from run import predict_grail_rank

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_size', type=int, default=64)
parser.add_argument('--num_attention_heads', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument("--task", type=str, default='fb237_v2')
parser.add_argument('--checkpoints', type=str, default='ckpts')
args = parser.parse_args()

save_path = os.path.join(args.checkpoints, args.task, 
                str(args.embedding_size)+'_'+str(args.learning_rate)+'_'+
                str(args.batch_size) +'_'+str(args.num_attention_heads)+'_'+"best_yet")
model = torch.load(save_path, map_location=device).to(device)
model.eval()


print('results for grail rank metrics')
eval_performance = predict_grail_rank(
        model=model, device=device, neg_path=args.neg_save_path_test)