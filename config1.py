import argparse

parser = argparse.ArgumentParser()


parser.add_argument(
  "--DDTtrain",
  default='/home/psdz/zhaoqi/retrieval/shangbiao/dataDDT/train2',
  help="The path of DDTmodel train.")

parser.add_argument(
  "--DDTimage",
  default='/home/psdz/zhaoqi/timg.jpeg',
  help="The path of DDTmodel test image.")

parser.add_argument(
  "--DDT_savedir",
  default='/home/psdz/zhaoqi/retrieval/shangbiao/dataDDT/result2',
  help="The path of DDTmodel DDT_savedir.")

args = parser.parse_args()

DDTtrain = str(args.DDTtrain)
DDTimage = str(args.DDTimage)
DDT_savedir = str(args.DDT_savedir)
