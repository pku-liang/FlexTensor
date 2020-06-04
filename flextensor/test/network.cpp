#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;
 
int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
 
void read_Mnist_Label(string filename, float labels[][10])
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;
		
	
		for (int i = 0; i < number_of_images; i++)
		{
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
            labels[i][label]=1;
		}
		
	}
}
 
void read_Mnist_Images(string filename, float images[][784])
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		unsigned char label;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		n_rows = ReverseInt(n_rows);
		n_cols = ReverseInt(n_cols);
 
		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;
		cout << "rows = " << n_rows << endl;
		cout << "cols = " << n_cols << endl;
 
		for (int i = 0; i < number_of_images; i++)
		{
			for (int r = 0; r < n_rows; r++)
			{
				for (int c = 0; c < n_cols; c++)
				{
					unsigned char image = 0;
                    file.read((char*)&image, sizeof(image));
                    images[i][r*n_cols+c]=(image-127)/128.0;
				}
			}
		}
	}
}
const int Size_train = 60000, Size_image = 784;
float train_images[Size_train][Size_image],tmp_images[Size_train][Size_image];
float train_labels[Size_train][10],tmp_labels[Size_train][10];
int id[Size_train];

const int Size_test=10000;
float test_images[Size_test][Size_image];
float test_labels[Size_test][10];

const int epochs = 20,batch_size=200;
const float lr = 0.00001;

void readtrain(){
    read_Mnist_Images("data/train-images.idx3-ubyte", train_images);
    read_Mnist_Label("data/train-labels.idx1-ubyte",train_labels);

    read_Mnist_Images("data/t10k-images.idx3-ubyte", test_images);
    read_Mnist_Label("data/t10k-labels.idx1-ubyte",test_labels);
}

#include <Halide.h>
#include <random>
#include <algorithm>
#include <cmath>
using namespace Halide;

float W0[10][144],B0[10],W1[10],cW[5][5];

void shuff(){
    int i,j;
    random_shuffle(id,id+Size_train);
    for(i=0;i<Size_train;i++){
        for(j=0;j<Size_image;j++)
            tmp_images[i][j]=train_images[id[i]][j];
        for(j=0;j<10;j++)
            tmp_labels[i][j]=train_labels[id[i]][j];
    }
    for(i=0;i<Size_train;i++){
        for(j=0;j<Size_image;j++)
            train_images[i][j]=tmp_images[i][j];
        for(j=0;j<10;j++)
            train_labels[i][j]=tmp_labels[i][j];
    }
}
int main(){

    std::default_random_engine e;
    std::normal_distribution<float> n(0, 0.1);
    for(int i=0;i<10;i++) B0[i]=0;
    for (int i=0;i<10;i++)
        for(int j=0;j<144;j++)
            W0[i][j]=n(e);

    for(int i=0;i<5;i++)
        for(int j=0;j<5;j++)
            cW[i][j]=n(e);

    Buffer<float> W((float*)W0,144,10);
    Buffer<float> B(B0,10);

    float pB=0;

    Buffer<float> conv_W((float*)cW,5,5);

    readtrain();

    for(int i=0;i<Size_train;i++) id[i]=i;
    MachineParams params(32, 16000000, 40);
    Target target = get_target_from_environment();

    for(int e=0;e<epochs;e++){
        shuff();
        for(int id=0;id<Size_train;id+=batch_size){
            Buffer<float> X(train_images[id],28,28,batch_size);
            Buffer<float> tgt(train_labels[id],10,batch_size);

            Func conv,pool,biased,flatten,Bb;
            Var x,y,i,z,p;
            RDom convr(0,5,0,5);

            //compute Y:
            //layer 1: conv
            Bb() = pB;
            conv(x,y,i) = 0.0f;
            conv(x,y,i) += conv_W(convr.x,convr.y)*X(x+convr.x,y+convr.y,i);
            pool(x,y,i) = Halide::max(conv(2*x,2*y,i),conv(2*x+1,2*y,i),conv(2*x,2*y+1,i),conv(2*x+1,2*y+1,i));
            biased(x,y,i) = tanh(pool(x,y,i)+Bb());
            flatten(z,i) = biased(z%12,z/12,i);

            //layer 2: fully-con
            Func ttY,tY,tmp,Y;
            RDom r1(0,144),r2(0,10);
            ttY(p,i) = B(p);
            ttY(p,i) += W(r1.x,p) * flatten(r1.x,i);
            tY(p,i) = exp(ttY(p,i));
            tmp(i) = 0.0f;
            tmp(i) += tY(r2.x,i);
            Y(p,i) = tY(p,i) / tmp(i);

            //compute loss:
            Func loss;
            RDom r3(0,10,0,batch_size);

            loss() = 0.0f;
            loss() += -Y(r3.x,r3.y)*log(tgt(r3.x,r3.y)+(float)(1e-5));

            //compute derivative:
            auto d = propagate_adjoints(loss);


            //backpropagate:
            Func newW,newB,newconvW,newpoolB;
            Var c;

            newW(z,p) = W(z,p) - d(W)(z,p)*lr;
            newB(p) = B(p) - d(B)(p)*lr;

            newconvW(x,y) = conv_W(x,y) - d(conv_W)(x,y)*lr;
            newpoolB() = pB - d(Bb)()*lr;
            //newpoolB(c) = - d(pool_B)(c);
            //newpoolB(c) = pool_B(c) - d(pool_B)(c)*lr;

            newW.set_estimate(z,0,144).set_estimate(p,0,10);
            newB.set_estimate(p,0,10);
            newconvW.set_estimate(x,0,5).set_estimate(y,0,5);
            //newpoolB.set_estimate(c,0,1);

            Pipeline({newW,newB,newconvW}).auto_schedule(target,params);

            //loss:
            Buffer<float> L = loss.realize();
            printf("loss in epoch %d batch %d: %f\n",e,id/batch_size,L(0));

            W = newW.realize(144,10);
            B = newB.realize(10);
            conv_W = newconvW.realize(5,5);
            Buffer<float> tmpb = newpoolB.realize();
            pB = tmpb(0);
            //pool_B = newpoolB.realize(1);
        }

        // eval:
        Buffer<float> X((float*)test_images,28,28,Size_test);
        Func conv,pool,biased,flatten;
        Var x,y,i,z,p;
        RDom convr(0,5,0,5),convc(0,1);

        //compute Y:
        //layer 1: conv
        conv(x,y,i) = 0.0f;
        conv(x,y,i) += conv_W(convr.x,convr.y)*X(x+convr.x,y+convr.y,i);
        pool(x,y,i) = Halide::max(conv(2*x,2*y,i),conv(2*x+1,2*y,i),conv(2*x,2*y+1,i),conv(2*x+1,2*y+1,i));
        biased(x,y,i) = tanh(pool(x,y,i));
        flatten(z,i) = biased(z%12,z/12,i);

        //layer 2: fully-con
        Func ttY,tY,tmp,Y;
        RDom r1(0,144),r2(0,10);
        ttY(p,i) = B(p);
        ttY(p,i) += W(r1.x,p) * flatten(r1.x,i);
        Buffer<float> res=ttY.realize(10,Size_test);
        int acc=0;
        for(int i=0;i<Size_test;i++){
            int tg=0;
            for(int j=0;j<10;j++)
                if(test_labels[i][j]>test_labels[i][tg]) tg=j;
            int my=0;
            for(int j=0;j<10;j++)
                if(res(j,i)>res(my,i)) my=j;
            if(tg==my) acc++;
        }
        printf("acc in epoch %d: %f\n",e,1.0*acc/Size_test);
    }
}
