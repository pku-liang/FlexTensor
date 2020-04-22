import tvm

A = tvm.te.placeholder([4, 4])

print("A requires_grad=", A.requires_grad)

B = tvm.te.placeholder([4, 4], requires_grad=True)

print("B requires_grad=", B.requires_grad)

C = tvm.te.compute([4, 4], lambda i, j: A[i, j])

print("C requires_grad=", C.requires_grad)

D = tvm.te.compute([4, 4], lambda i, j: A[i, j], requires_grad=True)

print("D requires_grad=", D.requires_grad)

E = tvm.te.compute([4, 4], lambda i, j: B[i, j])

print("E requires_grad=", E.requires_grad)

F = tvm.te.compute([4, 4], lambda i, j: B[i, j], requires_grad=True)

print("F requires_grad=", F.requires_grad)