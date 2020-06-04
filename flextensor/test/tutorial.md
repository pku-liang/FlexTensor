# Tutorial

### 1. Halide installation

​	Visit https://github.com/halide/Halide and follow README.md

​	See https://halide-lang.org/tutorials/tutorial_introduction.html for basic usage of Halide.

​	This tutorial serves as a supplement.

​	Include Halide.h first.

```c++
#include "Halide.h"

using namespace Halide;
```



## 2. Auto-scheduler

#### 2.1 Auto-scheduler with Halide generators

​	https://halide-lang.org/tutorials/tutorial_lesson_21_auto_scheduler_generate.html provides a tutorial for auto-scheduler with Halide generators. However, it's not very convenient.



#### 2.2 Simpler Usage of Auto-scheduler 

​	Firstly you should tell the auto-scheduler your machine parameters.

```c++
MachineParams params(32, 16000000, 40);
/*
	Halide::MachineParams is a struct representing the machine parameters to generate the auto-scheduled code for. It is defined as:
	MachineParams (int parallelism, uint64_t llc, float balance)
	parallelism: Maximum level of parallelism avalaible.
	llc: Size of the last-level cache (in bytes).
	balance: Indicates how much more expensive is the cost of a load compared to the cost of an arithmetic operation at last level cache.
*/
```

 	Then you should inform the auto-scheduler the OS it generates code for.

```c++
Target target=get_target_from_environment();
/*
	Or directly define it:
	Target target("x86-64-linux-sse41-avx-avx2");
*/
```

​	The last thing you should tell the auto-scheduler is your estimated range for each variables. 

```c++
f.set_estimate(x, 0, 1000)
/*
	f is a Halide::Func object
	set_estimate(var, min, extent) gives the estimated range for Variable var.
*/
```

​	Let's see a simple example.

```c++
Var x("x"), y("y");

Func f("f"), g("g"), h("h");
f(x, y) = (x + y) * (x + y);
g(x, y) = f(x, y) * 2 + 1;
h(x, y) = g(x, y) * 2 + 1;

h.set_estimate(x, 0, 1000).set_estimate(y, 0, 1000);
Pipeline(h).auto_schedule(target, params);
// h is auto-scheduled here. You can simply use h.realize(Xrange,Yrange).
```

​	If you want to schedule more than one Func:

```c++
Func f("f"), g("g"), h("h");
f(x, y) = (x + y) * (x + y);
g(x, y) = f(x, y) * 3 + 2;
h(x, y) = f(x, y) * 2 + 1;

g.set_estimate(x, 0, 1000).set_estimate(y, 0, 1000);
h.set_estimate(x, 0, 1000).set_estimate(y, 0, 1000);

Pipeline({g,h}).auto_schedule(target, params);
//g and h are auto-scheduled here.
```



### 3. Automatic differentiation in Halide

Halide also supports auto differentiation of some scalar $\mathcal{L}$.  Here is an basic example.

```c++
Func x("x");
x() = Expr(T(5));
Func y("y");

// define y = x^2 - 2x + 5 + 3 / x
y() = x() * x() - Expr(T(2)) * x() + Expr(T(5)) + Expr(T(3)) / x();

// compute derivative of y with respect to x
Derivative d = propagate_adjoints(y);
Func dx = d(x);
Buffer<T> dydx = dx.realize();

// we know that dydx = 2x - 2 - 3 / x^2 = 8 - 3 / 25, let's check it.
assert(fabs(dydx(0) - T(8.0 - 3.0 / 25.0)) < 1e-5);
```

Then we introduce a useful function 'select':

```c++
/*
	select is defined as follows:
	Expr Halide::select	(Expr condition,true_value,false_value)
    returns true_value when condition is true, otherwise returns false_value
*/
Func x("x");
x() = Expr(T(5));
Func y("y");

// y = 2x + 7x for x>0, 3x + 5x for x<0, 3x + 7x for x=0.
y() = select(x() > Expr(T(0)), Expr(T(2)) * x(), Expr(T(3)) * x()) +
      select(x() < Expr(T(0)), Expr(T(5)) * x(), Expr(T(7)) * x());

//compute derivative of y with respect to x, here we can see it equals 9.
Derivative d = propagate_adjoints(y);

assert(fabs(dydx(0) - T(9)) < 1e-5);
```

Note that we can compute derivative of some scalar $\mathcal{L}$ with respect to many variables, certainly it supports matrix or inputs in higher dimension.

```c++
// In the first part we define a loss function.
Param<float> g; // Gamma parameter
Buffer<float> im, tgt; // 2−D input and target buffers
Var x, y; // Integer variables for the pixel coordinates
Func f; // Halide function declarations

// Halide function definition
f(x, y) = pow(im(x, y), g);

// Reduction variables to loop over target's domain
RDom r(tgt);

Func loss; // We compute the MSE loss between f and tgt
loss() = 0.f; // Initialize the sum to 0

Expr diff = f(r.x, r.y) − tgt(r.x, r.y);
loss() += diff * diff; // Update definition

// In the second part we want to compute derivative of loss with respect to im and g.

// Obtain gradients with respect to image and gamma parameters
Derivative d_loss_d = propagate_adjoints(loss);
Func d_loss_d_g = d_loss_d(g);
Func d_loss_d_im = d_loss_d(im);

//Here we obtain derivative Func d_loss_d_g and d_loss_d_im. You can simply realize it.
```



### 4. Example: Train a Simple Network

We can use auto-scheduler and auto differentiation above to train a simple network.

```c++
//We omitted the initialization of parameters and data processing.
for(int e=0;e<epochs;e++){
    shuff();//SGD
    for(int id=0;id<Size_train;id+=batch_size){
        //Get MNIST dataset
        Buffer<float> X(train_images[id],28,28,batch_size);
        Buffer<float> tgt(train_labels[id],10,batch_size);
```



```c++
		Func conv,pool,biased,flatten,Bb;
		Var x,y,i,z,p;
        RDom convr(0,5,0,5);

        //layer 1: conv + maxpool
        Bb() = pB;// pool bias
        conv(x,y,i) = 0.0f;
        conv(x,y,i) += conv_W(convr.x,convr.y)*X(x+convr.x,y+convr.y,i);
        pool(x,y,i) = Halide::max(conv(2*x,2*y,i), conv(2*x+1,2*y,i), conv(2*x,2*y+1,i), conv(2*x+1,2*y+1,i));
        biased(x,y,i) = tanh(pool(x,y,i)+Bb());

		//flatten
        flatten(z,i) = biased(z%12,z/12,i);
```



```c++
        //layer 2: fully-connected + softmax
        Func ttY,tY,tmp,Y;
        RDom r1(0,144),r2(0,10);
		//ttY computes Matrix multiplication
        ttY(p,i) = B(p);
        ttY(p,i) += W(r1.x,p) * flatten(r1.x,i);
		//tY computes exp of tY
        tY(p,i) = exp(ttY(p,i));
		//normalization
        tmp(i) = 0.0f;
        tmp(i) += tY(r2.x,i);
        Y(p,i) = tY(p,i) / tmp(i);
```



```c++
        //compute loss:
        Func loss;
        RDom r3(0,10,0,batch_size);

		//Use Relative Entropy for loss:
        loss() = 0.0f;
        loss() += -Y(r3.x,r3.y)*log(tgt(r3.x,r3.y)+(float)(1e-5));
```



```c++
		//compute derivative:
        auto d = propagate_adjoints(loss);

        //back propagate:
        Func newW,newB,newconvW,newpoolB;
        Var c;

        newW(z,p) = W(z,p) - d(W)(z,p)*lr;
        newB(p) = B(p) - d(B)(p)*lr;

        newconvW(x,y) = conv_W(x,y) - d(conv_W)(x,y)*lr;
        newpoolB() = pB - d(Bb)()*lr;

        newW.set_estimate(z,0,144).set_estimate(p,0,10);
        newB.set_estimate(p,0,10);
        newconvW.set_estimate(x,0,5).set_estimate(y,0,5);

        Pipeline({newW,newB,newconvW}).auto_schedule(target,params);

        //loss:
        Buffer<float> L = loss.realize();
        printf("loss in epoch %d batch %d: %f\n",e,id/batch_size,L(0));

		//update parameters:
        W = newW.realize(144,10);
        B = newB.realize(10);
        conv_W = newconvW.realize(5,5);
        Buffer<float> tmpb = newpoolB.realize();
        pB = tmpb(0);
	}
}
```

Some tips about this training code:

1. Numerical processing in Halide is not so good as that in pytorch or tensorflow, so we need to carefully set hyperparameters.
2. There may be some warnings at auto scheduling. See https://github.com/halide/Halide/issues/3130 