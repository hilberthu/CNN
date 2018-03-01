#include <iostream>
#include <fstream>
#include <math.h>

using namespace std;

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}

#pragma once

#include "cnn.h"

CNN::CNN() {
    a = 1;
    r = 0;
    bool temp[6][16] = {
            {true,  false, false, false, true,  true,  true,  false, false, true,  true,  true,  true,  false, true,  true},
            {true,  true,  false, false, false, true,  true,  true,  false, false, true,  true,  true,  true,  false, true},
            {true,  true,  true,  false, false, false, true,  true,  true,  false, false, true,  false, true,  true,  true},
            {false, true,  true,  true,  false, false, true,  true,  true,  true,  false, false, true,  false, true,  true},
            {false, false, true,  true,  true,  false, false, true,  true,  true,  true,  false, true,  true,  false, true},
            {false, false, false, true,  true,  true,  false, false, true,  true,  true,  true,  false, true,  true,  true}
    };
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 16; ++j)
            connection[i][j] = temp[i][j];
}

CNN::~CNN() {
}

double CNN::random() {
    return rand() / (double) RAND_MAX;
}

double CNN::sigmoi(double x) {
    return 1.0 / (double) (1 + exp(-x));
}

double CNN::derivate_sigmoi(double z) {
    return z * (1 - z);
}

void CNN::init() {
    double scale = 1;
    cout << "init...";
    //timer.start();
    srand((unsigned int) time(NULL));
    //初始化C1层
    double fan_in = 1 * 5 * 5;
    double fan_out = 6 * 5 * 5;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 5; ++j)
            for (int k = 0; k < 5; ++k)
                C1_conv[i][j][k] = (random() - 0.5) * 2 / scale;
        C1_b[i] = 0;
    }

    //初始化C3层
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < 5; ++k)
                for (int l = 0; l < 5; ++l)
                    C3_conv[j][i][k][l] = (random() - 0.5) * 2 / scale;
        }
        C3_b[i] = 0;
    }

    //初始化C5层
    for (int i = 0; i < 120; ++i) {
        for (int j = 0; j < 16; ++j) {
            for (int k = 0; k < 5; ++k)
                for (int l = 0; l < 5; ++l)
                    C5_conv[j][i][k][l] = (random() - 0.5) * 2 / scale;
        }
        C5_b[i] = 0;
    }

    //初始化F6层
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 120; ++j)
            F6_p[j][i] = (random() - 0.5) * 2 / scale;
        F6_b[i] = 0;
    }
}

void CNN::readFile() {
    std::ifstream infile("train.csv");
    string header;
    int id, r, g, b;
    char comma;
    getline(infile, header);
    cout << "read train file...";
    //timer.start();
    for (int i = 0; i < train_size; ++i) {
        infile >> id;
        infile >> comma;
        infile >> label[i];
        for (int j = 0; j < 32; ++j) {
            for (int k = 0; k < 32; ++k) {
                infile >> comma;
                infile >> r;
                infile >> comma;
                infile >> g;
                infile >> comma;
                infile >> b;
                I0[i][j][k] = (r * 0.299 + g * 0.587 + b * 0.114) / 256; //灰度化输入图像，在缩放到0~1之间
                //cout<<I0[i][j][k]<<endl;
            }
        }
    }
    //cout<<"   cost time: "<<timer.end()<<'s'<<endl;


}

void CNN::ff(int index) {
    for (int i = 0; i < batch_size; ++i) {
        //C1层
        for (int j = 0; j < 6; ++j) {
            for (int x = 0; x < 28; ++x) {
                for (int y = 0; y < 28; ++y) {
                    double temp = 0;
                    for (int conv_x = 0; conv_x < 5; ++conv_x) {
                        for (int conv_y = 0; conv_y < 5; ++conv_y) {
                            temp += I0[index + i][x + conv_x][y + conv_y] * C1_conv[j][conv_x][conv_y];
                        }
                    }
                    C1[i][j][x][y] = sigmoi(temp + C1_b[j]);
                }
            }
        }

        //S2层
        for (int j = 0; j < 6; ++j) {
            for (int x = 0; x < 14; ++x) {
                for (int y = 0; y < 14; ++y) {
                    double temp = 0;
                    for (int conv_x = x * 2; conv_x < x * 2 + 2; ++conv_x) {
                        for (int conv_y = y * 2; conv_y < y * 2 + 2; ++conv_y) {
                            temp += C1[i][j][conv_x][conv_y];
                        }
                    }
                    S2[i][j][x][y] = temp / 4.0;
                }
            }
        }

        //C3层
        for (int j = 0; j < 16; ++j) //C3的map个数
        {
            for (int x = 0; x < 10; ++x) {
                for (int y = 0; y < 10; ++y) {
                    double temp = 0;
                    for (int k = 0; k < 6; ++k) //S2的map个数
                    {
                        if (connection[k][j]) {
                            for (int conv_x = 0; conv_x < 5; ++conv_x) {
                                for (int conv_y = 0; conv_y < 5; ++conv_y) {
                                    temp += S2[i][k][x + conv_x][y + conv_y] * C3_conv[k][j][conv_x][conv_y];
                                }
                            }
                        }

                    }
                    C3[i][j][x][y] = sigmoi(temp + C3_b[j]);
                }
            }
        }

        //S4层
        for (int j = 0; j < 16; ++j) {
            for (int x = 0; x < 5; ++x) {
                for (int y = 0; y < 5; ++y) {
                    double temp = 0;
                    for (int conv_x = x * 2; conv_x < x * 2 + 2; ++conv_x) {
                        for (int conv_y = y * 2; conv_y < y * 2 + 2; ++conv_y) {
                            temp += C3[i][j][conv_x][conv_y];
                        }
                    }
                    S4[i][j][x][y] = temp / 4.0;
                }
            }
        }

        //C5层
        for (int j = 0; j < 120; ++j) //C5map个数
        {
            double temp = 0;
            for (int k = 0; k < 16; ++k) {
                for (int conv_x = 0; conv_x < 5; ++conv_x) {
                    for (int conv_y = 0; conv_y < 5; ++conv_y) {
                        temp += S4[i][k][conv_x][conv_y] * C5_conv[k][j][conv_x][conv_y];
                    }
                }
            }
            C5[i][j] = sigmoi(temp + C5_b[j]);
        }

        //F6层
        for (int j = 0; j < 10; ++j) //F6神经元个数
        {
            double temp = 0;
            for (int k = 0; k < 120; ++k) {
                temp += C5[i][k] * F6_p[k][j];
            }
            F6[i][j] = exp(temp + F6_b[j]);
        }
    }
}

void CNN::bp(int index) {
    for (int i = 0; i < batch_size; ++i) {

        double sum = 0;
        for (int j = 0; j < 10; ++j) {
            sum += F6[i][j];
        }
        for (int j = 0; j < 10; ++j) {
            if (label[index + i] == j)
                F6_d[i][j] = F6[i][j] / sum - 1;
            else
                F6_d[i][j] = F6[i][j] / sum;
        }

        //C5层
        for (int j = 0; j < 120; ++j) {
            double temp = 0;
            for (int k = 0; k < 84; ++k) {
                temp += F6_d[i][k] * F6_p[j][k];
            }
            C5_d[i][j] = temp * derivate_sigmoi(C5[i][j]);
        }

        //S4层
        for (int j = 0; j < 16; ++j) {
            for (int x = 0; x < 5; ++x) {
                for (int y = 0; y < 5; ++y) {
                    double temp = 0;
                    for (int k = 0; k < 120; ++k) {
                        temp += C5_d[i][k] * C5_conv[j][k][x][y];
                    }
                    //S4_d[i][j][x][y] = temp * derivate_sigmoi(S4[i][j][x][y]);
                    S4_d[i][j][x][y] = temp;
                }
            }
        }

        //C3层
        for (int j = 0; j < 16; ++j) {
            for (int x = 0; x < 10; ++x) {
                for (int y = 0; y < 10; ++y) {
                    C3_d[i][j][x][y] = derivate_sigmoi(C3[i][j][x][y]) * (S4_d[i][j][x / 2][y / 2] / 4.0);
                    //C3_d[i][j][x][y] = derivate_sigmoi(C3[i][j][x][y]) * S4_d[i][j][x / 2][y / 2] * S4_w[j] / 4.0;
                }
            }
        }

        //S2层
        for (int j = 0; j < 6; ++j) {
            for (int x = 0; x < 14; ++x) {
                for (int y = 0; y < 14; ++y) {
                    double temp = 0;
                    int min_x = x - 4 < 0 ? 0 : x - 4;
                    int max_x = x + 4 > 13 ? 13 : x + 4;
                    int min_y = y - 4 < 0 ? 0 : y - 4;
                    int max_y = y + 4 > 13 ? 13 : y + 4;
                    for (int k = 0; k < 16; ++k) {
                        if (connection[j][k]) {
                            for (int l = min_x; l <= max_x - 4; ++l) {
                                for (int m = min_y; m <= max_y - 4; ++m) {
                                    temp += C3_d[i][k][l][m] * C3_conv[j][k][x - l][y - m];
                                }
                            }
                        }
                    }
                    //S2_d[i][j][x][y] = temp * derivate_sigmoi(S2[i][j][x][y]);
                    S2_d[i][j][x][y] = temp;
                }
            }
        }

        //C1层
        for (int j = 0; j < 6; ++j) {
            for (int x = 0; x < 28; ++x) {
                for (int y = 0; y < 28; ++y) {
                    //C1_d[i][j][x][y] = derivate_sigmoi(C1[i][j][x][y]) * S2_d[i][j][x / 2][y / 2] * S2_w[j] / 4.0;
                    //cout<<S2_d[i][j][x / 2][y / 2] / 4.0<<" ";
                    C1_d[i][j][x][y] = derivate_sigmoi(C1[i][j][x][y]) * S2_d[i][j][x / 2][y / 2] / 4.0;
                }
                //cout<<endl;
            }
        }
    }

    for (int i = 0; i < 120; ++i) {
        for (int j = 0; j < 10; ++j) {
            double temp = 0;
            for (int k = 0; k < batch_size; ++k) {
                temp += F6_d[k][j] * C5[k][i];
            }
            //cout<<temp / batch_size<<endl;
            //cout<<f7_p[i][j]<<"=>";
            F6_dp[i][j] = temp / batch_size;
            //cout<<f7_p[i][j]<<endl;
        }
    }
    //更新bias
    for (int i = 0; i < 10; ++i) {
        double temp = 0;
        for (int k = 0; k < batch_size; ++k) {
            temp += F6_d[k][i];
        }
        F6_db[i] = temp / batch_size;
    }

    //C5层
    //更新卷积核
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 120; ++j) {
            double temp[5][5] = {{0},
                                 {0}};

            for (int x = 0; x < 5; ++x) {
                for (int y = 0; y < 5; ++y) {
                    for (int k = 0; k < batch_size; ++k) {
                        temp[x][y] += S4[k][i][x][y] * C5_d[k][j];
                    }
                }
            }
            for (int x = 0; x < 5; ++x) {
                for (int y = 0; y < 5; ++y) {
                    C5_dconv[i][j][x][y] = temp[x][y] / batch_size;
                }
            }
        }
    }
    //更新bias
    for (int i = 0; i < 120; ++i) {
        double temp = 0;
        for (int k = 0; k < batch_size; ++k) {
            temp += C5_d[k][i];
        }
        C5_db[i] = temp / batch_size;
    }

    //C3层
    //更新卷积核
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 16; ++j) {
            if (connection[i][j]) {
                double temp[5][5] = {{0},
                                     {0}};
                for (int k = 0; k < batch_size; ++k) {
                    for (int x = 0; x < 10; ++x) {
                        for (int y = 0; y < 10; ++y) {
                            for (int conv_x = 0; conv_x < 5; ++conv_x) {
                                for (int conv_y = 0; conv_y < 5; ++conv_y) {
                                    temp[conv_x][conv_y] += C3_d[k][j][x][y] * S2[k][i][x + conv_x][y + conv_y];
                                }
                            }
                        }
                    }
                }
                for (int x = 0; x < 5; ++x) {
                    for (int y = 0; y < 5; ++y) {
                        C3_dconv[i][j][x][y] = temp[x][y] / batch_size;
                    }
                }
            }
        }
    }
    //更新bias
    for (int i = 0; i < 16; ++i) {
        double temp = 0;
        for (int k = 0; k < batch_size; ++k) {
            for (int x = 0; x < 10; ++x) {
                for (int y = 0; y < 10; ++y) {
                    temp += C3_d[k][i][x][y];
                }
            }
        }
        C3_db[i] = temp / batch_size;
    }

    //C1层
    //更新卷积核
    for (int i = 0; i < 6; ++i) {
        double temp[5][5] = {{0},
                             {0}};
        for (int k = 0; k < batch_size; ++k) {
            for (int x = 0; x < 28; ++x) {
                for (int y = 0; y < 28; ++y) {
                    for (int conv_x = 0; conv_x < 5; ++conv_x) {
                        for (int conv_y = 0; conv_y < 5; ++conv_y) {
                            temp[conv_x][conv_y] += C1_d[k][i][x][y] * I0[index + k][x + conv_x][y + conv_y];
                        }
                    }
                }
            }
        }
        for (int x = 0; x < 5; ++x) {
            for (int y = 0; y < 5; ++y) {
                C1_dconv[i][x][y] = temp[x][y] / batch_size;
                //cout<<temp[x][y]<<" ";
            }
            //cout<<endl;
        }

    }
    //cout<<C1_conv[0][0][0]<<endl;
    //更新bias
    for (int i = 0; i < 6; ++i) {
        double temp = 0;
        for (int k = 0; k < batch_size; ++k) {
            for (int x = 0; x < 28; ++x) {
                for (int y = 0; y < 28; ++y) {
                    temp += C1_d[k][i][x][y];
                }
            }
        }
        //cout<<C1_b[i]<<"=>";
        C1_db[i] = temp / batch_size;
        //if(i == 0)
        //cout<<a * temp / batch_size<<endl;
        //cout<<C1_b[i]<<endl;

    }
}

void CNN::update(int index) {

    //F6层
    //更新权重
    for (int i = 0; i < 120; ++i) {
        for (int j = 0; j < 10; ++j) {
            /*double temp = 0;
            for(int k = 0; k < batch_size; ++k)
            {
                temp += F6_d[k][j] * C5[k][i];
            }*/
            F6_p[i][j] -= a * (F6_dp[i][j] + r * F6_p[i][j]);
        }
    }
    //更新bias
    for (int i = 0; i < 10; ++i) {
        /* double temp = 0;
       for(int k = 0; k < batch_size; ++k)
       {
           temp += F6_d[k][i];
       }*/
        F6_b[i] -= a * F6_db[i];
    }

    //C5层
    //更新卷积核
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 120; ++j) {
            for (int x = 0; x < 5; ++x) {
                for (int y = 0; y < 5; ++y) {
                    C5_conv[i][j][x][y] -= a * (C5_dconv[i][j][x][y] + r * C5_conv[i][j][x][y]);
                }
            }
        }
    }
    //更新bias
    for (int i = 0; i < 120; ++i) {
        C5_b[i] -= a * C5_db[i];
    }


    //C3层
    //更新卷积核
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 16; ++j) {
            if (connection[i][j]) {
                for (int x = 0; x < 5; ++x) {
                    for (int y = 0; y < 5; ++y) {
                        C3_conv[i][j][x][y] -= a * (C3_dconv[i][j][x][y] + r * C3_conv[i][j][x][y]);
                    }
                }
            }
        }
    }
    //更新bias
    for (int i = 0; i < 16; ++i) {
        C3_b[i] -= a * C3_db[i];
    }

    //C1层
    //更新卷积核
    for (int i = 0; i < 6; ++i) {
        for (int x = 0; x < 5; ++x) {
            for (int y = 0; y < 5; ++y) {
                C1_conv[i][x][y] -= a * (C1_dconv[i][x][y] + r * C1_conv[i][x][y]);
                //cout<<temp[x][y]<<" ";
            }
            //cout<<endl;
        }

    }
    //cout<<C1_conv[0][0][0]<<endl;
    //更新bias
    for (int i = 0; i < 6; ++i) {
        C1_b[i] -= a * C1_db[i];
    }


}

void CNN::train() {
    cout << "train..." << endl;
    //timer.start();
    double pre_ar = 0, cur_ar;
    int batch_num = train_size / batch_size;
    for (int i = 0; i < train_times; ++i) {
        for (int j = 0; j < batch_num; ++j) {
            ff(j * batch_size);
            bp(j * batch_size);
            update(j * batch_size);
            //evaluate();
        }
        a *= 0.85;
        a = max(0.00001, a);
        cur_ar = evaluate();
        if (pre_ar != 0)
            if (pre_ar > cur_ar) break;
        pre_ar = cur_ar;
    }
}

double CNN::evaluate() {
    int batch_num = train_size / batch_size;
    int right = 0;
    for (int i = 0; i < batch_num; ++i) {
        ff(i * batch_size);
        for (int j = 0; j < batch_size; ++j) {
            double max = F6[j][0];
            int pre = 0;
            for (int k = 1; k < 10; ++k) {
                /*if(i == 0 && j == 0){
                    cout<<F7[j][k]<<" ";
                }*/
                if (F6[j][k] > max) {
                    max = F6[j][k];
                    pre = k;
                }
            }
            if (label[i * batch_size + j] == pre)
                ++right;
        }
    }
    cout << right << " " << train_size << endl;
    cout << "AR:" << right / (double) train_size << endl;
    return right / (double) train_size;
}

void CNN::test() {
    ifstream infile("test.csv");
    ofstream outfile("result.csv");
    string header;
    int id, r, g, b;
    char comma;
    getline(infile, header);
    cout << "read test file...";
    //timer.start();
    for (int i = 0; i < test_size; ++i) {
        infile >> id;
        for (int j = 0; j < 32; ++j) {
            for (int k = 0; k < 32; ++k) {
                infile >> comma;
                infile >> r;
                infile >> comma;
                infile >> g;
                infile >> comma;
                infile >> b;
                testI0[i][j][k] = (r * 0.299 + g * 0.587 + b * 0.114) / 256; //灰度化输入图像，在缩放到0~1之间
            }
        }
    }
    infile.close();
    //cout << "   cost time:" << timer.end() << "s" << endl;

    cout << "predict...";
    //timer.start();
    outfile << "id,label" << endl;
    for (int i = 0; i < test_size; ++i) {
        //C1层
        for (int j = 0; j < 6; ++j) {
            for (int x = 0; x < 28; ++x) {
                for (int y = 0; y < 28; ++y) {
                    double temp = 0;
                    for (int conv_x = 0; conv_x < 5; ++conv_x) {
                        for (int conv_y = 0; conv_y < 5; ++conv_y) {
                            temp += testI0[i][x + conv_x][y + conv_y] * C1_conv[j][conv_x][conv_y];
                        }
                    }
                    C1[0][j][x][y] = sigmoi(temp + C1_b[j]);
                }
            }
        }

        //S2层
        for (int j = 0; j < 6; ++j) {
            for (int x = 0; x < 14; ++x) {
                for (int y = 0; y < 14; ++y) {
                    double temp = 0;
                    for (int conv_x = x * 2; conv_x < x * 2 + 2; ++conv_x) {
                        for (int conv_y = y * 2; conv_y < y * 2 + 2; ++conv_y) {
                            temp += C1[0][j][conv_x][conv_y];
                        }
                    }
                    S2[0][j][x][y] = temp / 4.0;
                }
            }
        }

        //C3层
        for (int j = 0; j < 16; ++j) //C3的map个数
        {
            for (int x = 0; x < 10; ++x) {
                for (int y = 0; y < 10; ++y) {
                    double temp = 0;
                    for (int k = 0; k < 6; ++k) //S2的map个数
                    {
                        for (int conv_x = 0; conv_x < 5; ++conv_x) {
                            for (int conv_y = 0; conv_y < 5; ++conv_y) {
                                temp += S2[0][k][x + conv_x][y + conv_y] * C3_conv[k][j][conv_x][conv_y];
                            }
                        }
                    }
                    C3[0][j][x][y] = sigmoi(temp + C3_b[j]);
                }
            }
        }

        //S4层
        for (int j = 0; j < 16; ++j) {
            for (int x = 0; x < 5; ++x) {
                for (int y = 0; y < 5; ++y) {
                    double temp = 0;
                    for (int conv_x = x * 2; conv_x < x * 2 + 2; ++conv_x) {
                        for (int conv_y = y * 2; conv_y < y * 2 + 2; ++conv_y) {
                            temp += C3[0][j][conv_x][conv_y];
                        }
                    }
                    S4[0][j][x][y] = temp / 4.0;
                }
            }
        }

        //C5层
        for (int j = 0; j < 120; ++j) //C5map个数
        {
            double temp = 0;
            for (int k = 0; k < 16; ++k) {
                for (int conv_x = 0; conv_x < 5; ++conv_x) {
                    for (int conv_y = 0; conv_y < 5; ++conv_y) {
                        temp += S4[0][k][conv_x][conv_y] * C5_conv[k][j][conv_x][conv_y];
                    }
                }
            }
            C5[0][j] = sigmoi(temp + C5_b[j]);
        }

        //F6层
        for (int j = 0; j < 10; ++j) //F6神经元个数
        {
            double temp = 0;
            for (int k = 0; k < 120; ++k) {
                temp += C5[0][k] * F6_p[k][j];
            }
            F6[0][j] = exp(temp + F6_b[j]);
        }

        double max = -1;
        int pre = -1;
        for (int j = 0; j < 10; ++j) {
            if (F6[0][j] > max) {
                max = F6[0][j];
                pre = j;
            }
        }
        outfile << i + 1 << "," << pre << endl;
    }
    outfile.close();
}

double CNN::getCost_checkgGrad() {
    double sum = 0;
    for (int i = 0; i < 10; ++i) {
        sum += F6[0][i];
    }
    double temp = 0;
    for (int i = 0; i < 10; ++i) {
        if (label[0] == i)
            temp = -log(F6[0][i] / sum);
    }
    return temp;
}

void CNN::checkGrad() {
    readFile();
    init();

    double epsilon = pow(10, -4);
    double er = pow(10, -8);

    double temp, add, sub, d, e;

    cout << "check.." << endl;

    //F6
    cout << "F6...";
    for (int i = 0; i < 120; ++i) {
        for (int j = 0; j < 10; ++j) {
            temp = F6_p[i][j];

            F6_p[i][j] = temp + epsilon;
            ff(0);
            add = getCost_checkgGrad();

            F6_p[i][j] = temp - epsilon;
            ff(0);
            sub = getCost_checkgGrad();

            F6_p[i][j] = temp;
            ff(0);
            bp(0);
            d = (add - sub) / (epsilon * 2);
            e = fabs(d - F6_dp[i][j]);
            if (e > er) {
                cout << "F6_p[" << i << "][" << j << "]" << " numerical gradient checking failed" << endl;
            }
        }
    }
    for (int i = 0; i < 10; ++i) {
        temp = F6_b[i];

        F6_b[i] = temp + epsilon;
        ff(0);
        add = getCost_checkgGrad();

        F6_b[i] = temp - epsilon;
        ff(0);
        sub = getCost_checkgGrad();

        F6_b[i] = temp;
        ff(0);
        bp(0);
        d = (add - sub) / (epsilon * 2);
        e = fabs(d - F6_db[i]);
        if (e > er) {
            cout << "F6_b[" << i << "]" << " numerical gradient checking failed" << endl;
        }

    }
    cout << "over" << endl;

    //C5层
    cout << "C5...";
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 120; ++j) {
            for (int x = 0; x < 5; ++x) {
                for (int y = 0; y < 5; ++y) {
                    temp = C5_conv[i][j][x][y];

                    C5_conv[i][j][x][y] = temp + epsilon;
                    ff(0);
                    add = getCost_checkgGrad();

                    C5_conv[i][j][x][y] = temp - epsilon;
                    ff(0);
                    sub = getCost_checkgGrad();

                    C5_conv[i][j][x][y] = temp;
                    ff(0);
                    bp(0);

                    d = (add - sub) / (epsilon * 2);
                    e = fabs(d - C5_dconv[i][j][x][y]);
                    if (e > er) {
                        cout << "C5_conv[" << i << "][" << j << "][" << x << "][" << y << "]"
                             << " numerical gradient checking failed" << endl;
                    }
                }
            }
        }
    }
    for (int i = 0; i < 120; ++i) {
        temp = C5_b[i];

        C5_b[i] = temp + epsilon;
        ff(0);
        add = getCost_checkgGrad();

        C5_b[i] = temp - epsilon;
        ff(0);
        sub = getCost_checkgGrad();

        C5_b[i] = temp;
        ff(0);
        bp(0);
        d = (add - sub) / (epsilon * 2);
        e = fabs(d - C5_db[i]);
        if (e > er) {
            cout << "C5_b[" << i << "]" << " numerical gradient checking failed" << endl;
        }
    }
    cout << "over" << endl;
    //C3
    cout << "C3...";
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 16; ++j) {
            if (connection[i][j]) {
                for (int x = 0; x < 5; ++x) {
                    for (int y = 0; y < 5; ++y) {
                        temp = C3_conv[i][j][x][y];

                        C3_conv[i][j][x][y] = temp + epsilon;
                        ff(0);
                        add = getCost_checkgGrad();

                        C3_conv[i][j][x][y] = temp - epsilon;
                        ff(0);
                        sub = getCost_checkgGrad();

                        C3_conv[i][j][x][y] = temp;
                        ff(0);
                        bp(0);

                        d = (add - sub) / (epsilon * 2);
                        e = fabs(d - C3_dconv[i][j][x][y]);
                        //if(i == 0 && j== 0)
                        //{
                        //  cout << "d = " << d <<endl;
                        //  cout << "C3_dconv[i][j][x][y]" << C3_dconv[i][j][x][y] << endl;
                        //}
                        if (e > er) {
                            cout << "C3_conv[" << i << "][" << j << "][" << x << "][" << y << "]"
                                 << " numerical gradient checking failed" << endl;
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < 16; ++i) {
        temp = C3_b[i];

        C3_b[i] = temp + epsilon;
        ff(0);
        add = getCost_checkgGrad();

        C3_b[i] = temp - epsilon;
        ff(0);
        sub = getCost_checkgGrad();

        C3_b[i] = temp;
        ff(0);
        bp(0);
        d = (add - sub) / (epsilon * 2);
        e = fabs(d - C3_db[i]);

        if (e > er) {
            cout << "C3_b[" << i << "]" << " numerical gradient checking failed" << endl;
        }
    }
    cout << "over" << endl;

    //C1
    cout << "C1...";
    for (int j = 0; j < 6; ++j) {
        for (int x = 0; x < 5; ++x) {
            for (int y = 0; y < 5; ++y) {
                temp = C1_conv[j][x][y];

                C1_conv[j][x][y] = temp + epsilon;
                ff(0);
                add = getCost_checkgGrad();

                C1_conv[j][x][y] = temp - epsilon;
                ff(0);
                sub = getCost_checkgGrad();

                C1_conv[j][x][y] = temp;
                ff(0);
                bp(0);

                d = (add - sub) / (epsilon * 2);
                e = fabs(d - C1_dconv[j][x][y]);
                if (e > er) {
                    cout << "C1_conv[" << j << "][" << x << "][" << y << "]" << " numerical gradient checking failed"
                         << endl;
                    cout << "d = " << d << endl;
                    cout << "C1_dconv[" << j << "][" << x << "][" << y << "] = " << C1_dconv[j][x][y] << endl;
                    cout << "add = " << add << "  sub = " << sub << endl;
                }
            }
        }
    }
    for (int i = 0; i < 6; ++i) {
        temp = C1_b[i];

        C1_b[i] = temp + epsilon;
        ff(0);
        add = getCost_checkgGrad();

        C1_b[i] = temp - epsilon;
        ff(0);
        sub = getCost_checkgGrad();

        C1_b[i] = temp;
        ff(0);
        bp(0);
        d = (add - sub) / (epsilon * 2);
        e = fabs(d - C1_db[i]);
        if (e > er) {
            cout << "C1_b[" << i << "]" << " numerical gradient checking failed" << endl;
            cout << "d = " << d << endl;
            cout << "C1_db[" << i << "] = " << C1_db[i] << endl;
        }
    }
    cout << "over" << endl;
    //cout << "cost time:" << timer.end() << endl;
}