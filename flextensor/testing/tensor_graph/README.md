# Tensor Graph: Using GNN to optimize Tensor Operators

1. Prerequsiities:
   - Python >= 3.5
   - PyTorch >= 1.2.0
   - [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
   - [TVM >= 0.6.0](https://docs.tvm.ai/install/from_source.html)
   - [FlexTensor](https://github.com/KnowingNothing/FlexTensor.git)

2. Run:
   `python train.py --help` to see optional knobs

3. Use trained model on Titan X:
   `python train.py --only_test --fmodel gemm_model/gemm_model.pkl --ftest dataset/gemm_test.txt --eval_dev 0`

4. The dataset:

   GEMM: gemm_train.txt, gemm_test.txt

   Conv2d: conv2d_train.txt, conv2d_test.txt

5. Any problems:
   File issues to https://github.com/KnowingNothing/FlexTensor.git
   Tensor Graph is a testing feature of FlexTensor currently.