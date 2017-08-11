#include <cstdio>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <string>
#include <ctime>
#include <random>

using namespace std;

enum
{
	Conv,
	Pool
};

struct convNet_2D
{
	typedef vector <double> mat;
	typedef vector <mat>	mat2D;

	struct MultilayerPerceptron
	{
		vector <double> output;
		vector <int> & numOfUnits;
		vector < vector < vector <double> > > weights;
		vector < vector <double> > units;
		vector < vector <double> > in;
		vector <double> neededOut;
		vector <int> isThereBias;
		double learningRate;
		int numOfLayer;
		vector < vector <double> > delta;

		// _numOfUnits : Layer 별 unit의 수 ex) { 3,4,2 } 이면 입력은 3개, 아웃풋은 2개 히든 레이어 유닛은 4개
		// path : 가중치를 저장해놓은 파일이 있는 주소
		MultilayerPerceptron(vector <int> & _numOfUnits, string path = "") :numOfUnits(_numOfUnits)
		{
			numOfLayer = _numOfUnits.size();

			for (int i = 0; i < numOfLayer - 1; i++)isThereBias.push_back(1);
			isThereBias.push_back(0);

			weights.resize(numOfLayer - 1);
			for (int i = 1; i < numOfLayer; i++)
			{
				weights[i - 1].resize(_numOfUnits[i]);
				for (int j = 0; j < weights[i - 1].size(); j++)
					weights[i - 1][j].resize(_numOfUnits[i - 1] + isThereBias[i - 1]);
			}

			// 가중치를 저장한 파일의 경로가 있으면 불러오고 아니면 랜덤으로 수를 할당
			if (path == "")
				initWeights();
			else
				loadWeights(path);
		}

		void fit(const vector <double>& xTrain, const vector <double> & yTrain, double _learningRate, const vector <double> & init_in)
		{
			learningRate = _learningRate;

			neededOut = yTrain;
			units.clear();
			units.resize(numOfLayer);
			in.clear();
			in.resize(numOfLayer);
			in[0] = init_in;

			for (int j = 0; j < xTrain.size(); j++)
				units[0].push_back(xTrain[j]);

			if (isThereBias[0])
				units[0].push_back(1.0);

			Learning();
		}

		void getOutput(const vector <double> & xTest)
		{
			units.clear();
			units.resize(numOfLayer);
			in.clear();
			in.resize(numOfLayer);

			for (int i = 0; i < xTest.size(); i++) units[0].push_back(xTest[i]);
			if (isThereBias[0])units[0].push_back(1.0);

			forward(units);
			output.clear();

			for (int i = 0; i < units[numOfLayer - 1].size(); i++) output.push_back(units[numOfLayer - 1][i]);
		}

		void saveWeights(const string & path)
		{
			FILE * fp = fopen(path.c_str(), "w");

			for (int i = 0; i < weights.size(); i++)
				for (int j = 0; j < weights[i].size(); j++)
					for (int k = 0; k < weights[i][j].size(); k++)
						fprintf(fp, "%lf ", weights[i][j][k]);

			fclose(fp);
		}

		void loadWeights(const string & path)
		{
			FILE * fp = fopen(path.c_str(), "r");

			for (int i = 0; i < weights.size(); i++)
				for (int j = 0; j < weights[i].size(); j++)
					for (int k = 0; k < weights[i][j].size(); k++)
						fscanf(fp, "%lf", &weights[i][j][k]);

			fclose(fp);
		}

		void initWeights()
		{
			default_random_engine generator;
			normal_distribution<double> distribution(0.0, 0.01);

			for (int i = 0; i < weights.size(); i++)
				for (int j = 0; j < weights[i].size(); j++)
					for (int k = 0; k < weights[i][j].size(); k++)
						weights[i][j][k] = distribution(generator);
		}

		double sigmoid(const double & x)
		{
			return 1 / (1 + exp(-x));
		}

		double relu(const double & x)
		{
			return max(x, 0.0);
		}

		double dRelu(const double & x)
		{
			if (x > 0)return 1.0;
			else return 0.0;
		}

		double dSigmoid(const double & x)
		{
			double tmp = sigmoid(x);
			return (1 - tmp)*tmp;
		}

		double getInnerProduct(const vector <double> & W, const vector <double> & X)
		{
			double ret = 0;
			for (int i = 0; i < W.size(); i++)ret += W[i] * X[i];
			return ret;
		}

		void calcOneLayer(vector <double> & out, const vector < vector <double> > & W, const vector <double> & input, bool isThereBias, int curLayer)
		{
			in[curLayer].resize(W.size());
			for (int i = 0; i < W.size(); i++)
			{
				in[curLayer][i] = getInnerProduct(W[i], input);
				double tmp = (curLayer == numOfLayer - 1) ? in[curLayer][i] : relu(in[curLayer][i]);
				out.push_back(tmp);
			}

			if (isThereBias)out.push_back(1.0);
		}

		void forward(vector < vector <double> > & _units)
		{			
			for (int i = 1; i < numOfLayer; i++)
				calcOneLayer(_units[i], weights[i - 1], _units[i - 1], isThereBias[i], i);
			

			double sum = 0;
			for (int i = 0; i < _units[numOfLayer - 1].size(); i++)
				sum += exp(in[numOfLayer - 1][i]);

			for (int i = 0; i < _units[numOfLayer - 1].size(); i++)
				_units[numOfLayer - 1][i] = exp(in[numOfLayer - 1][i]) / sum;
		}

		void backward()
		{
			delta.clear();
			delta.resize(numOfLayer);

			// 초기 기울기 값을 생성
			for (int i = 0; i < units[numOfLayer - 1].size(); i++)
				delta[numOfLayer - 1].push_back(units[numOfLayer - 1][i] * (1 - units[numOfLayer - 1][i])
					*(neededOut[i] - units[numOfLayer - 1][i]));

			for (int l = numOfLayer - 2; l >= 0; l--)
			{
				for (int j = 0; j < units[l].size(); j++)
				{
					if (j < units[l].size() - 1 || (isThereBias[l] == 0 && j == units[l].size() - 1))
					{
						// 기울기값인 delta 를 생성
						double tmp = 0;
						for (int i = 0; i < units[l + 1].size() - isThereBias[l + 1]; i++)
							tmp += weights[l][i][j] * delta[l + 1][i];
						if (l != 0) delta[l].push_back(dRelu(in[l][j])*tmp);
						else delta[l].push_back(0);
					}
					// 가중치 업데이트
					for (int i = 0; i < units[l + 1].size() - isThereBias[l + 1]; i++)
						weights[l][i][j] += learningRate*units[l][j] * delta[l + 1][i];
				}
			}
		}

		void Learning()
		{
			forward(units);
			backward();
		}
	};
	
	vector < MultilayerPerceptron > mv;

	convNet_2D(int input_num, int output_num, const string & mlp_path = "")
	{
		int MLP_hidden_layers = 0;
		int unit, i = 0;

		vector <int> nu = {};	//Setting Layer Unit

		printf("What number do you want to have MLP Hidden Layers?\n");
		scanf("%d", &MLP_hidden_layers);
		nu.push_back(input_num);
		for (i = 0; i < MLP_hidden_layers; i++)
		{
			printf("What number do you want to have MLP %d hidden layer units?\n", i + 1);
			scanf("%d", &unit);
			nu.push_back(unit);
		}
		nu.push_back(output_num);

		MultilayerPerceptron mlp(nu, mlp_path);

		mv.push_back(mlp);
	}

	void trainXY(const mat & trainX, const mat & trainY, double _learningRate)
	{
		mat in0;
		mv[0].fit(trainX, trainY, _learningRate, in0);
	}

	void getOutput(const mat & testX)
	{
		mv[0].getOutput(testX);
	}
};

vector < vector <double> > testX;
vector <int> desiredY;
vector < vector <double> > trainX;
vector < vector <double> > trainY;
int mxvl = 0;

void readTest(int input_size, int test_num)
{
	FILE * fp = fopen("test.txt", "r");
	testX.resize(test_num, vector <double>(input_size, 0.0));
	desiredY.resize(test_num);

	int y = -1;
	int x = -1;
	int cnt = -1;
	int s;

	while (~fscanf(fp, "%d", &s))
	{
		x++;
		cnt++;
		if (cnt % (input_size + 1) == 0)
		{
			if (y == test_num - 1)break;
			y++;
			desiredY[y] = s;

			x = -1;
			continue;
		}

		testX[y][x] = (double)s;
	}
	fclose(fp);
}

void readTrain(int input_size, int output_size, int train_num)
{
	FILE * fp = fopen("train.txt", "r");

	trainX.resize(train_num, vector <double>(input_size, 0.0));
	trainY.resize(train_num, vector <double>(output_size, 0.0));

	int y = -1;
	int x = -1;
	int cnt = -1;
	int s;

	while (~fscanf(fp, "%d", &s))
	{
		x++;
		cnt++;
		if (cnt % (input_size + 1) == 0)
		{
			if (y == train_num - 1)break;
			y++;
			trainY[y][s] = 1.0;
			x = -1;
			continue;
		}

		trainX[y][x] = (double)s;
	}

	fclose(fp);
}

void test(convNet_2D & convNet, int test_num)
{
	clock_t st = clock();
	int cn = 0;

	for (int i = 0; i < test_num; i++)
	{
		convNet.getOutput(testX[i]);

		double mxvl = -1e9;
		int indx = -1;

		for (int j = 0; j < 10; j++)
		{
			if (convNet.mv[0].output[j] > mxvl)
			{
				mxvl = convNet.mv[0].output[j];
				indx = j;
			}
		}
		if (indx == desiredY[i])cn++;
	}

	if (mxvl < cn)
	{
		mxvl = cn;

		convNet.mv[0].saveWeights("bestMvWeights.txt");
	}

	printf("duration : %d ms\n", clock() - st);
	printf("correct ratio : %d/%d best ratio : %d/%d\n", cn, test_num, mxvl, test_num);
}

void learning(convNet_2D & convNet, int cnt1, int train_num, double learning_rate)
{
	//	puts("before save ");
	convNet.mv[0].saveWeights("prev_mvWeights.txt");

	clock_t st = clock();

	for (int j = 0; j < 1; j++)
	{
		for (int i = 0; i < train_num; i++)
		{
			if (i % 1000 == 0)printf("j : %d i : %d\n", j, i);
			convNet.trainXY(trainX[i], trainY[i], learning_rate);
		}
	}

	printf("duration : %d ms\n", clock() - st);


	convNet.mv[0].saveWeights("mvWeights.txt");

	if (cnt1 != -1)
	{
		char mvs[50], cvs[50];

		sprintf(mvs, "mvWeights%d.txt", cnt1);

		string ms = mvs;

		//	puts("before save ");
		convNet.mv[0].saveWeights(ms);
	}
}

void conv(int input_size, int output_size, int train_num, int test_num)
{
	convNet_2D convNet(input_size, output_size);
	//convNet_2D convNet(input_size, output_size, "mvWeights.txt");
	double learning_rate;
	readTest(input_size, test_num);
	readTrain(input_size, output_size, train_num);

	printf("Choose Learning rate for Training.\n");
	scanf("%lf", &learning_rate);
	for(int cnt = 0; cnt < 100; cnt++)
	{
	printf("cnt : %d\n", cnt);
	learning(convNet, -1, train_num, learning_rate);
	printf("start test\n");
	test(convNet, test_num);
	}
	//test(convNet, test_num);

	puts("end");
}

int main()
{
	int input_size, output_size = 0;
	int train_number, test_number = 0;

	printf("What is the number of training inputs?\n");
	scanf("%d", &train_number);
	printf("What is the number of test inputs?\n");
	scanf("%d", &test_number);
	printf("What is the input layer size?\n");
	scanf("%d", &input_size);
	printf("What is the output layer size?\n");
	scanf("%d", &output_size);
	
	conv(input_size, output_size, train_number, test_number);

	while (1);
	return 0;
}
