# Tensor Graph: Using GNN to optimize Tensor Operators

1. Prerequsiities:
   - Python >= 3.5
   - PyTorch >= 1.2.0
   - [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
   - [TVM >= 0.6.0](https://docs.tvm.ai/install/from_source.html)
   - [AutoScheduler](https://github.com/KnowingNothing/AutoScheduler.git)

2. Run:
   `python train.py --help` to see optional knobs

3. Use trained model on Titan X:
   `python train.py --only_test --fmodel first-run-20191224/20191224.pkl --ftest dataset/gemm_test.txt --eval_dev 0`

4. The dataset:
   GEMM: gemm_train.txt, gemm_test.txt
   Conv2d: conv2d_train.txt, conv2d_test.txt

5. Any problems:
   Fire issues to https://github.com/KnowingNothing/AutoScheduler.git
   Tensor Graph is a testing feature of AutoScheduler currently.