#include "mainwindow.h"
#include <QApplication>
#include <QCoreApplication>
#include <iostream>
#include <vector>
#include <cmath>
#include <QDir>
#include <QFileInfoList>
#include <QStringList>
#include <string>
#include <QDebug>



struct ParkingTraj
{
  float reward;

  float forward_s_length = 1.0;
  float reverse_s_length = 1.0;
  float forward_turn_length = 1.0;
  float reverse_turn_length = 1.0;
  float end_s_length;
  float likehood;
  std::vector<float> feature_set;  //forward length, back length,forward curve path, reverse path,
};

QString TextStreamRead(QString path)
{
    QString rData;
    QFile file(path);
    if(file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QByteArray line = file.readAll();
        rData=line;
    }
    file.close();
    return rData;
}


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

   int n_iters = 3000; // Set the desired value for n_iters
   int feature_num = 4; // Set the desired value for feature_num
   double beta1 = 0.9; // Set the desired value for beta1
   double beta2 = 0.999; // Set the desired value for beta2
   double eps = 1e-8; // Set the desired value for eps
   double lam = 0.01; // Set the desired value for lam
   double lr = 0.1; // Set the desired value for lr

   std::vector<std::vector<ParkingTraj>> train_buffer; // Define the train_buffer data structure
   std::vector<double> weight(feature_num, -1.0); // Initialize the theta vector
   std::vector<float> max_train_buffer; // Define the train_buffer data structure
   weight[0] = -1.0;
   weight[1] = -1.0;
   weight[2] = -1.0;
   weight[3] = -1.0;

   std::vector<std::vector<double>> iter_weight;
   iter_weight.push_back(weight);

   // Update weights
   std::vector<double> pm(feature_num, 0.0);
   std::vector<double> pv(feature_num, 0.0);

   // The last traj in one scene must be the ground truth
   QString folderPath = "/home/jiaotong404/planning/human-like-parking-trajectory-planning/plan_simluation/simulation/qtPro-TruckSim-2.5/plan_simulation/build-TruckSim-Desktop_Qt_5_10_1_GCC_64bit-Debug/train_data/";
   QDir folder(folderPath);
   QStringList lpxFiles;

   if (folder.exists())
   {
       QFileInfoList fileList = folder.entryInfoList(QStringList() << "*.txt", QDir::Files);
       for (const QFileInfo& fileInfo : fileList) {
           lpxFiles.append(fileInfo.fileName());
       }
   } else {
       qDebug() << "文件夹不存在";
   }

   for(int k = 0; k < lpxFiles.size(); ++k)
   {
       QString filename = folderPath+lpxFiles.at(k);
       QString main_path_data = TextStreamRead(filename);
       QStringList row_datas = main_path_data.split("\n");
       std::vector<ParkingTraj> scene_traj;
       float max_forward_s = 1.0;
       float max_reverse_s = 1.0;
       float max_forward_turn = 1.0;
       float max_reverse_turn = 1.0;
       float max_length = 1.0;

       for(int i = 0; i < row_datas.size() - 1 && row_datas.size() > 1; i++)
       {
           ParkingTraj temp_traj;
           QString row_data = row_datas.at(i);
           QStringList row_data_sprit = row_data.split(",");

           std::vector<std::pair<QString,double>> seg_curve;
           seg_curve.push_back(std::make_pair(row_data_sprit.at(4), row_data_sprit.at(3).toDouble()));
           seg_curve.push_back(std::make_pair(row_data_sprit.at(6), row_data_sprit.at(5).toDouble()));
           seg_curve.push_back(std::make_pair(row_data_sprit.at(8), row_data_sprit.at(7).toDouble()));
           if(row_data_sprit.back().toDouble() < 1.0)
               continue;

           if (row_data_sprit.size() > 10)
           {
               seg_curve.push_back(std::make_pair(row_data_sprit.at(10),row_data_sprit.at(9).toDouble()));
           }

           if (row_data_sprit.size() > 12)
           {
               seg_curve.push_back(std::make_pair(row_data_sprit.at(12),row_data_sprit.at(11).toDouble()));
           }

           for(int k = 0 ; k < seg_curve.size();++k)
           {
               if(seg_curve[k].first != "S")
               {
                   if(seg_curve[k].second > 0)
                   {
                       temp_traj.forward_turn_length += fabs(seg_curve[k].second);
                   }else
                   {
                       temp_traj.reverse_turn_length += fabs(seg_curve[k].second);
                   }
               } else
               {
                   if(seg_curve[k].second > 0 )
                   {
                       temp_traj.forward_s_length += fabs(seg_curve[k].second);
                   }else
                   {
                       temp_traj.reverse_s_length += fabs(seg_curve[k].second);
                   }
               }
           }

           max_forward_s = std::max(max_forward_s, temp_traj.forward_s_length);
           max_reverse_s = std::max(max_reverse_s, temp_traj.reverse_s_length);
           max_forward_turn = std::max(max_forward_turn, temp_traj.forward_turn_length);
           max_reverse_turn = std::max(max_reverse_turn, temp_traj.reverse_turn_length);
           scene_traj.push_back(temp_traj);
       }

       max_length = std::max(max_length, max_forward_s);
       max_length = std::max(max_length, max_reverse_s);
       max_length = std::max(max_length, max_forward_turn);
       max_length = std::max(max_length, max_reverse_turn);

       max_train_buffer.push_back(max_length);
       train_buffer.push_back(scene_traj);
   }

   for(int i = 0; i < train_buffer.size(); ++i)
   {
        for(int j = 0; j < train_buffer[i].size();++j)
        {
            ParkingTraj temp_traj = train_buffer[i][j];
            train_buffer[i][j].feature_set.push_back(temp_traj.forward_s_length/max_train_buffer[i]*10.0);
            train_buffer[i][j].feature_set.push_back(temp_traj.reverse_s_length/max_train_buffer[i]*10.0);
            train_buffer[i][j].feature_set.push_back(temp_traj.forward_turn_length/max_train_buffer[i]*10.0);
            train_buffer[i][j].feature_set.push_back(temp_traj.reverse_turn_length/max_train_buffer[i]*10.0);
            float likeness = 0.0;
            likeness = pow((temp_traj.forward_s_length - train_buffer[i].back().forward_s_length)/max_train_buffer[i], 2.0)+
                    pow((temp_traj.reverse_s_length - train_buffer[i].back().reverse_s_length)/max_train_buffer[i], 2.0)+
                    pow((temp_traj.forward_turn_length - train_buffer[i].back().forward_turn_length)/max_train_buffer[i], 2.0)+
                    pow((temp_traj.reverse_turn_length - train_buffer[i].back().reverse_turn_length)/max_train_buffer[i], 2.0);
            train_buffer[i][j].likehood = sqrt(likeness);
        }
   }

   std::vector<double> iteration_human_likeness;  // Initialize the iteration_human_likeness vector
   iteration_human_likeness.push_back(100000);
   std::vector<double> human_like_train_accuracy;

   for (int iteration = 0; iteration < n_iters; iteration++)
   {
       std::vector<double> feature_exp(feature_num, 0.0); // Initialize the feature_exp vector
       std::vector<double> human_feature_exp(feature_num, 0.0); // Initialize the human_feature_exp vector
       double human_likeness_vals = 0.0;

       // Fix collision feature's weight
       int index = 0;
       float log_prob = 0.0;
       for (auto& scene : train_buffer)
       {
           for (auto& trajectory : scene)
           {
               double reward = 0.0;
               for (int i = 0; i < feature_num; i++) {
                   reward += (trajectory.feature_set[i] * weight[i]);
               }

               trajectory.reward = reward;
           }

           // Calculate probability of each trajectory
           std::vector<double> rewards;
           for (auto& traj : scene)
           {
               rewards.push_back(traj.reward);
           }

           std::vector<double> probs(rewards.size());
           double sum_probs = 0.0;

           for (int i = 0; i < rewards.size(); i++) {
               probs[i] = std::exp(rewards[i]);
               sum_probs += probs[i];
           }

           for (int i = 0; i < probs.size(); i++) {
               probs[i] /= sum_probs;
           }
           log_prob += std::log10(probs.back());
           int maxIndex = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
           std::vector<double> max_index_probs(rewards.size(),0.0);
           max_index_probs[maxIndex] = 1.0;

//           qDebug()<<"max index: "<<maxIndex<<" ground truth:"<<probs.size()-1;

           // Calculate feature expectation with respect to the weights
           int feature_num = 4;
           for (int i = 0; i < scene.size(); i++)
           {
               for (int j = 0; j < feature_num; j++)
               {
                   double sum_features =  probs[i]* scene[i].feature_set[j];
                   feature_exp[j] += sum_features;
               }
           }

           // Select trajectories to calculate human likeness
           std::vector<int> sorted_indices(probs.size());
           for (int i = 0; i < probs.size(); i++) {
               sorted_indices[i] = i;
           }

           std::sort(sorted_indices.begin(), sorted_indices.end(),
               [&](int a, int b) { return probs[a] > probs[b]; });

           if(scene.size() > 1)
           {
               float likehood = 100000.0;
               for(int i = 0; i < 2; ++i)
               {
                   int idx = sorted_indices[i];
                   likehood = std::min(likehood, scene[idx].likehood);
               }

               if(iteration == n_iters-1)
               {
                   human_like_train_accuracy.push_back(likehood);
               }
               human_likeness_vals += likehood;
           }

           // Calculate human trajectory feature
           for (int i = 0; i < feature_num; i++) {
               human_feature_exp[i] += scene.back().feature_set[i];
           }

           // Go to next trajectory
           index++;
       }

       iteration_human_likeness.push_back(human_likeness_vals);

       // Compute gradient
       float feature_difference = 0.0;
       std::vector<double> grad(feature_num, 0.0);
       for (int i = 0; i < feature_num; i++)
       {
           grad[i] = human_feature_exp[i] - feature_exp[i] - 2 * lam * weight[i];
           feature_difference += grad[i];
       }

       for (int i = 0; i < grad.size(); i++) {
           pm[i] = beta1 * pm[i] + (1 - beta1) * grad[i];
           pv[i] = beta2 * pv[i] + (1 - beta2) * (grad[i] * grad[i]);
       }


       for (int i = 0; i < grad.size(); i++) {
           double mhat = pm[i] / (1 - std::pow(beta1, iteration + 1));
           double vhat = pv[i] / (1 - std::pow(beta2, iteration + 1));
           double update_vec = mhat / (std::sqrt(vhat) + eps);
           weight[i] += lr * update_vec;
       }

       if(iteration % 1 == 0)
       {
           std::vector<double> weight_temp{weight[0], weight[1], weight[2], weight[3]};
           iter_weight.push_back(weight_temp);
       }
   }


   for(int i =0; i <human_like_train_accuracy.size(); ++i)
   {
       std::cout<<"Trainaccuracy: "<<human_like_train_accuracy.at(i)<<std::endl;
   }

   // ====================== for  test data ===============================

   // The last traj in one scene must be the ground truth
   QString testfolderPath = "/home/jiaotong404/planning/human-like-parking-trajectory-planning/plan_simluation/simulation/qtPro-TruckSim-2.5/plan_simulation/build-TruckSim-Desktop_Qt_5_10_1_GCC_64bit-Debug/test_data/";
   QDir test_folder(testfolderPath);
   QStringList test_lpxFiles;
   std::vector<float> max_test_buffer; // Define the train_buffer data structure
   std::vector<std::vector<ParkingTraj>> test_buffer;

   if (test_folder.exists()) {
       QFileInfoList fileList = test_folder.entryInfoList(QStringList() << "*.txt", QDir::Files);
       for (const QFileInfo& fileInfo : fileList)
       {
           test_lpxFiles.append(fileInfo.fileName());
       }
   } else
   {
       qDebug() << "文件夹不存在";
   }

   for(int k = 0; k < test_lpxFiles.size(); ++k)
   {
       QString filename = testfolderPath+test_lpxFiles.at(k);
       QString main_path_data = TextStreamRead(filename);
       QStringList row_datas = main_path_data.split("\n");
       std::vector<ParkingTraj> scene_traj;
       float max_forward_s = 1.0;
       float max_reverse_s = 1.0;
       float max_forward_turn = 1.0;
       float max_reverse_turn = 1.0;
       float max_length = 1.0;

       for(int i = 0; i < row_datas.size() - 1 && row_datas.size() > 1; i++)
       {
           ParkingTraj temp_traj;
           QString row_data = row_datas.at(i);
           QStringList row_data_sprit = row_data.split(",");

           std::vector<std::pair<QString,double>> seg_curve;
           seg_curve.push_back(std::make_pair(row_data_sprit.at(4), row_data_sprit.at(3).toDouble()));
           seg_curve.push_back(std::make_pair(row_data_sprit.at(6), row_data_sprit.at(5).toDouble()));
           seg_curve.push_back(std::make_pair(row_data_sprit.at(8), row_data_sprit.at(7).toDouble()));
           if(row_data_sprit.back().toDouble() < 1.0)
               continue;

           if (row_data_sprit.size() > 10)
           {
               seg_curve.push_back(std::make_pair(row_data_sprit.at(10),row_data_sprit.at(9).toDouble()));
           }

           if (row_data_sprit.size() > 12)
           {
               seg_curve.push_back(std::make_pair(row_data_sprit.at(12),row_data_sprit.at(11).toDouble()));
           }

           for(int k = 0 ; k< seg_curve.size();++k)
           {
               if(seg_curve[k].first != "S")
               {
                   if(seg_curve[k].second > 0)
                   {
                       temp_traj.forward_turn_length += fabs(seg_curve[k].second);
                   }else
                   {
                       temp_traj.reverse_turn_length += fabs(seg_curve[k].second);
                   }
               } else
               {
                   if(seg_curve[k].second > 0 )
                   {
                       temp_traj.forward_s_length += fabs(seg_curve[k].second);
                   }else
                   {
                       temp_traj.reverse_s_length += fabs(seg_curve[k].second);
                   }
               }
           }

           max_forward_s = std::max(max_forward_s, temp_traj.forward_s_length);
           max_reverse_s = std::max(max_reverse_s, temp_traj.reverse_s_length);
           max_forward_turn = std::max(max_forward_turn, temp_traj.forward_turn_length);
           max_reverse_turn = std::max(max_reverse_turn,temp_traj.reverse_turn_length);
           scene_traj.push_back(temp_traj);
       }

       max_length = std::max(max_length,max_forward_s);
       max_length = std::max(max_length,max_reverse_s);
       max_length = std::max(max_length,max_forward_turn);
       max_length = std::max(max_length,max_reverse_turn);

       max_test_buffer.push_back(max_length);
       test_buffer.push_back(scene_traj);
   }



   for(int i = 0; i < test_buffer.size(); ++i)
   {
        for(int j = 0; j < test_buffer[i].size();++j)
        {
            ParkingTraj temp_traj = test_buffer[i][j];
            test_buffer[i][j].feature_set.push_back(temp_traj.forward_s_length/max_test_buffer[i]*10.0);
            test_buffer[i][j].feature_set.push_back(temp_traj.reverse_s_length/max_test_buffer[i]*10.0);
            test_buffer[i][j].feature_set.push_back(temp_traj.forward_turn_length/max_test_buffer[i]*10.0);
            test_buffer[i][j].feature_set.push_back(temp_traj.reverse_turn_length/max_test_buffer[i]*10.0);
            float likeness = 0.0;
            likeness = pow((temp_traj.forward_s_length - test_buffer[i].back().forward_s_length)/max_test_buffer[i], 2.0)+
                    pow((temp_traj.reverse_s_length - test_buffer[i].back().reverse_s_length)/max_test_buffer[i], 2.0)+
                    pow((temp_traj.forward_turn_length - test_buffer[i].back().forward_turn_length)/max_test_buffer[i], 2.0)+
                    pow((temp_traj.reverse_turn_length - test_buffer[i].back().reverse_turn_length)/max_test_buffer[i], 2.0);
            test_buffer[i][j].likehood = sqrt(likeness);
        }
   }

   std::vector<double> test_accuracy;

   for(int k = 0; k< iter_weight.size(); ++k)
   {
       float truth_num = 0.0;
       for(int i = 0; i < test_buffer.size();++i)
       {
           double optimal_cost= -10000.0;
           int optimal_index = -1;
           int sub_optimal_index = -1;
           for(int j=0; j< test_buffer[i].size(); ++j)
           {
                test_buffer[i][j].reward = (test_buffer[i][j].forward_s_length*iter_weight[k][0]+ test_buffer[i][j].reverse_s_length*iter_weight[k][1]+
                         test_buffer[i][j].forward_turn_length*iter_weight[k][2] + test_buffer[i][j].reverse_turn_length*iter_weight[k][3])/max_test_buffer[i]*10.0;

                if(test_buffer[i][j].reward > optimal_cost)
                {
                    optimal_cost = test_buffer[i][j].reward;
                    sub_optimal_index = optimal_index;
                    optimal_index = j;
                }
           }
           if(k == iter_weight.size()-1)
           {
               test_accuracy.push_back(test_buffer[i][optimal_index].likehood);
           }

           if(test_buffer[i].size() >3)
           {
               if(optimal_index == test_buffer[i].size() - 1 || sub_optimal_index == test_buffer[i].size() - 1)
                   truth_num += 1.0;
           }else
           {
               if(optimal_index == test_buffer[i].size() - 1)
                   truth_num += 1.0;
           }

       }
//       qDebug()<<" ============  test accuracy: "<< truth_num/test_buffer.size() <<" ===================";
   }

   for(int i =0; i <test_accuracy.size(); ++i)
   {
       std::cout<<"Testaccuracy: "<<test_accuracy.at(i)<<std::endl;
   }


    return a.exec();
}
