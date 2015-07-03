//============================================================================
// Name        : eventGabor.cpp
// Author      : yeshasvi tvs
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#if defined MAX_RES
#else
#define MAX_RES 128
#endif

//TODO Need to redo the GPU Implementation details with Phase shift details
#define NUMFILTERS 2 //Number of ST Filters, Blocks in the GPU Kernel
#define NUMDIRECTIONS 8 //Number of Directions, Threads in each block of GPU Kernel
#define NUMPHASES 7 //Number of Phases

#include <cuda.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <cmath>
#include <tuple>
#include <numeric>
#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <yarp/conf/options.h>
#include <iCub/emorph/all.h>

#include <opencv/cxcore.h>
#include <opencv2/core/core.hpp>
#include <opencv/highgui.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "eventHistoryBuffer.hpp"
#include "stFilters.h"
#include "stFiltersGPU.h"


using namespace yarp::os;
using namespace emorph;
using namespace std;
using namespace cv;

//CUDA Kernel for GPU Computation

//_global__ void convolutionKernerl(){

//}


//Global variables

//Event History buffer instance for LEFT and RIGHT Eyes
eventHistoryBuffer event_history_left;
eventHistoryBuffer event_history_right;
int kernel_size;
double temporal_diff_step;


//Number of directions
int number_directions;
double theta_step;
double max_theta;
int theta_index;
std::vector<double> theta_vector;
std::vector<double>::iterator theta_vector_it;

//Number of Phases
int number_phases;
double phase_step;
double max_phase;
int phase_index;
std::vector<double> phase_vector;
std::vector<double>::iterator phase_vector_it;

//Disparity Vectors
std::vector<double> disparity_vector;
std::vector<double>::iterator disparity_vector_it;

//Disparity Estimate vectors
//std::vector< std::vector<double> > disparity_est_theta(NUMDIRECTIONS,std::vector<double>(NUMPHASES,0)); //Two dimensional vector to store both directions and phases - Each row for each theta and each column for each phase
std::vector<double> disparity_est_theta;
std::vector<double>::iterator disparity_est_theta_it;


emorph::vtsHelper unwrap;
double event_time;


double even_conv_left,odd_conv_left;
//NOTE Check if this can be implmented in any other ways
double even_conv_right[NUMPHASES];
double odd_conv_right[NUMPHASES]; //Arrays of size = number of phases
double final_convolution_even[NUMPHASES];
double final_convolution_odd[NUMPHASES];
double final_convolution[NUMPHASES];


std::vector<double> binocular_enery_theta;
std::vector<double>::iterator binocular_enery_theta_it;
void motionHypothesis(int&,int&);

//Quiver function
void quiver_plot(int&, int&, double&,double&);

//Instance of stFitlering class with filter parameters as arguments
stFilters st_filters;//Default parameters

double eta; //Normalizing parameter for MT pooling


//Output files
std::vector<string> files;
std::string flow_file = "flow.txt";
ofstream ffile;

void initialize(){

    kernel_size = 11;
    temporal_diff_step = 1;


    //Number of directions - Changing the range from 0 to PI, NOTE
    max_theta =  M_PI;
    number_directions = 8;
    theta_step = 0.125;

    //Number of phases - In the range of - PI to PI
    max_phase = M_PI;
    number_phases = NUMPHASES; //Check this once again
    phase_step = (2*M_PI)/7; //Check this once again

    eta = 0.0001; //TODO  check the value

    //Checking the Values of empty arrays
    /*for(int t=0;t<8;t++){

        std::cout<< "EVEN RIGHT VALUES : " << even_conv_right[t]<<std::endl;
        std::cout<< "ODD RIGHT VALUES : " << odd_conv_right[t]<<std::endl;

    }*/


}


//Final Disparity Estimation
void disparity_estimation(){

    theta_vector_it = theta_vector.begin();
    disparity_est_theta_it = disparity_est_theta.begin();
    double disp_x_theta[NUMDIRECTIONS];
    double disp_y_theta[NUMDIRECTIONS];
    std::fill_n(disp_x_theta,NUMDIRECTIONS,0); //Initializing with zeroes
    std::fill_n(disp_y_theta,NUMDIRECTIONS,0);
    double sumX = 0;
    double sumY = 0;
    double sumXX = 0;
    double sumYY = 0;
    double sumXY = 0;

    //Final Disparity Values Estimated
    double disp_x = 0;
    double disp_y= 0;

    theta_index=0;
    for(; theta_vector_it!= theta_vector.end() ;){ //Going over the size of number of directions

        disp_x_theta[theta_index] = *disparity_est_theta_it * cos(*theta_vector_it);
        disp_y_theta[theta_index] = *disparity_est_theta_it * sin(*theta_vector_it);
        sumXX = sumXX + (cos(*theta_vector_it * *theta_vector_it) * cos(*theta_vector_it * *theta_vector_it));
        sumYY = sumYY + (sin(*theta_vector_it * *theta_vector_it) * sin(*theta_vector_it * *theta_vector_it));
        sumXY = sumXY + (cos(*theta_vector_it * *theta_vector_it) * sin(*theta_vector_it * *theta_vector_it));

        ++theta_vector_it;
        ++disparity_est_theta_it;
        ++theta_index;

    }

    theta_vector_it = theta_vector.begin();
    theta_index=0;
    for(; theta_vector_it!= theta_vector.end() ;){

        sumX = sumX + disp_x_theta[theta_index];
        sumY = sumY + disp_y_theta[theta_index];
        ++theta_vector_it;
        ++theta_index;
    }

    //Computing the final disparity values

    disp_x = ( sumYY*sumX - sumXY*sumY )/(sumXX*sumYY-sumXY*sumXY);
    disp_y = ( sumXX*sumY - sumXY*sumX )/(sumXX*sumYY-sumXY*sumXY);

    std::cout << "Estimated Final Disparity Values : " << disp_x << " " <<disp_y<<std::endl;//Debug Code

    //Clearing the disparity estimation vector
    disparity_est_theta.clear();




}



int main(int argc, char *argv[])
{


    initialize(); //Initializing the values

   //Resource finder 
   yarp::os::ResourceFinder rf;
   rf.configure(argc,argv);
	
   //Set up yarp network
   yarp::os::Network yarp;

   //Variables
   yarp::os::BufferedPort<emorph::vBottle> inputPort;
   yarp::os::BufferedPort<emorph::vBottle> outputPort;


   emorph::vBottle *tempEvents; //Temporary events
   

   //Using Resource finder for port names
   string inputPortName = rf.find("inputPort").asString();
   string outputPortName = rf.find("outputPort").asString();    
   string sourcePortName = rf.find("sourcePort").asString();

   //Setting default port names
   if(inputPortName == ""){ inputPortName = "/inputEvents"; }
   if(outputPortName == ""){ outputPortName = "/outputEvents"; }
   if(sourcePortName == ""){ sourcePortName = "/aexGrabber/vBottle:o"; }

   //Port connections
   bool ok = inputPort.open(inputPortName.c_str()); //Opening input port
   ok = ok && outputPort.open(outputPortName.c_str()); //Opening output port

   
   //checking ports
   if (!ok) {
    fprintf(stderr, "Failed to create ports.\n");
    fprintf(stderr, "Maybe you need to start a nameserver (run 'yarpserver')\n");
    return 1;
   }

   if(yarp.connect(sourcePortName.c_str(), inputPortName.c_str(), "tcp")){
 	  
    std::cout << "source port to input port connection successful... " <<std::endl;

   }
   else{std::cout << "source port to input port connection failed! " <<std::endl;}

   //Setting the theta vector based on the number of directios
   double value = 0;
   for(int p=1; p<=number_directions ;p++){ //Loop runs for number of directions

       //value = value + theta_step; //Line here in case of testing with single direction i.e theta
       theta_vector.push_back( value * M_PI);
       value = value + theta_step; //Line here in case all directions and with fixed theta step

   }

   //Setting the phase vector based on the number of phases
   double phase_gen[]={-1,0.5,-0.33,0,0.33,0.5,1};
   double phase_value=0; //Local Varialbe
   for(int t=0; t< number_phases; t++){

       //std::cout << "Phase Gen :"<< phase_gen[t] <<std::endl; //Debug Code
       phase_value = max_phase * phase_gen[t];

       phase_vector.push_back(phase_value);
       //std::cout<<"Phase Values : "<< phase_value <<std::endl; //Debug Code

       disparity_vector.push_back(phase_value/st_filters.omega);
       //std::cout<<"Disparity Values : "<< phase_value/st_filters.omega <<std::endl; //Debug Code

   }



   while(true){ //TODO wait till the input port receives data and run


    tempEvents = inputPort.read(true); //Read the incoming events into a temporary vBottle
    //std::cout << "received bottle" << std::endl;
    //continue;

    emorph::vQueue q; //Event queue
   	emorph::vQueue::iterator q_it;//Event queue iterator
 	
    tempEvents->getAll(q); // 1 ms timeframe
    //std::cout << "Processing " << q.size()<< " events" <<std::endl;

 	for (q_it = q.begin(); q_it != q.end(); q_it++) //Processing each event
            {

                emorph::AddressEvent *aep = (*q_it)->getAs<emorph::AddressEvent>(); //Getting each event from the Queue
                if(!aep) continue; //If AER is not received continue
                int channel = aep->getChannel(); //Processing events of only both channel - LEFT and Right Eye


                if(channel==0){

                    //Updating the event history buffer - LEFT Eye
                    event_history_left.updateList(*aep);

                }
                else if(channel==1){

                    //Updating the event history buffer - RIGHT Eye
                    event_history_right.updateList(*aep);


                }

                    //NOTE : Sensor X and Y are reversed
                    int pos_x = aep->getY();
                    int pos_y = aep->getX();
                    int current_event_polarity = aep->getPolarity();
                    if(current_event_polarity == 0){current_event_polarity = -1;} //Changing polarity to negative

                    //std::cout<< "Event at "<<"Channel: "<<channel<< " X : "<< pos_x <<" Y : "<< pos_y << " Polarity : "<<current_event_polarity<<std::endl; //Debug code


                    //Time stamp of the recent event
                    event_time = unwrap (aep->getStamp());

                    //converting to seconds
                    event_time = event_time / event_history_left.time_scale;


                    //Setting the filter spatial center values - Both Left and Right Filters are centered at the same spatial location - SPATIAL CORRELATION
                    st_filters.center_x = pos_x;
                    st_filters.center_y = pos_y;

                    //Processing for each direction i.e Theta
                    //std::cout << "Number of Directions : " << theta_vector.size()<<std::endl;//Debug Code
                    theta_vector_it = theta_vector.begin();
                    theta_index = 0;

                    for(; theta_vector_it != theta_vector.end() ; ){


                        //Open a file to write values for each ditection
                        stringstream itos;
                        itos << theta_index;

                        string file_name;
                        file_name = 'd'+ itos.str() + ".txt";
                        files.push_back(file_name);

                        ofstream dfile;
                        string file = files.at(theta_index);
                        dfile.open(file.c_str(),ios::in | ios::app);

                        //NOTE : The Following code is DEBUG
                        string full_file;
                        full_file = "full.txt";

                        ofstream fufile;
                        fufile.open(full_file.c_str(),ios::in | ios::app);


                        double theta = *theta_vector_it;
                        //std::cout<<"Processing for Theta : " << theta<<std::endl;//Debug Code

                        //Resetting filter convolution value for every event processing
                        even_conv_left = 0;
                        odd_conv_left = 0;

                        //Right Eye convolution variables are stored in the array of size equal to number of phases
                        std::fill_n(even_conv_right,number_phases,0);
                        std::fill_n(odd_conv_right,number_phases,0);
                        std::fill_n(final_convolution_even,number_phases,0);
                        std::fill_n(final_convolution_odd,number_phases,0);
                        std::fill_n(final_convolution,number_phases,0);

                        //Spatial and Temporal neighborhood processing - Left Eye
                        for(int j=1 ; j<= kernel_size ; j++){
                            for(int i=1 ; i <= kernel_size ; i++){

                                //Pixels to be processed in the spatial neighbourhood for both eyes - SPATIAL CORRELATION
                                int pixel_x = pos_x + i-6;
                                int pixel_y = pos_y + j-6;
                                double temporal_difference;

                                if(pixel_x >= 0 && pixel_y>= 0 && pixel_x < MAX_RES && pixel_y < MAX_RES){  //Checking for borders


                                    //std::cout << "Event Processing Left Buffer..."<<std::endl;//Debug Code
                                    //LEFT EYE PROCESSING

                                    event_history_left.timeStampsList_it = event_history_left.timeStampList[pixel_x][pixel_y].rbegin(); //Going from latest event pushed in the back

                                    //NOTE : List size is always limited to the buffer size, took care of this in the event buffer code
                                    if(!event_history_left.timeStampList[pixel_x][pixel_y].empty()){ //If the list is empty skip computing

                                        //std::cout << "Buffer Size Left Eye : " <<  event_history_left.timeStampList[pixel_x][pixel_y].size() << std::endl;//Debug Code

                                        for( int list_length = 1 ; list_length <= event_history_left.timeStampList[pixel_x][pixel_y].size() ; ){


                                            //NOTE : Timestamps in the list are encoded with polarity information
                                             temporal_difference = event_time - abs(*event_history_left.timeStampsList_it); //The first value is always zero
                                             //std::cout << "Temporal difference : " << temporal_difference <<std::endl; //Debug code

                                             //Here the neighborhood processing goes over 0 - 1 time scale
                                              if(temporal_difference > temporal_diff_step){
                                                //std::cout << "Skipping the list element..." << std::endl;//Debug Code
                                                ++list_length;
                                                continue;
                                             }


                                             double phase=0.0; //Local Variable
                                             //Call the spatio-temporal filtering function
                                             std::pair<double,double> conv_value = st_filters.filtering(pixel_x, pixel_y, theta, temporal_difference,phase);

                                             //TODO Check the effects of Polarity here! Without polarity its fine to get just the motion energy
                                             even_conv_left = even_conv_left +  conv_value.first;
                                             odd_conv_left = odd_conv_left +   conv_value.second;

                                             ++event_history_left.timeStampsList_it; //Moving up the list
                                             ++list_length;



                                        }//End of temporal iteration loop

                                    }
                                    else {

                                        //std::cout << "Skipping empty list of Left Eye..." << std::endl; //Debug Code
                                        continue;
                                    }     

                          }


                           else{

                                 //std::cout<< "Pixels Out of Bordes...."<<std::endl; //Debug Code
                                 continue;
                               }

                           }
                        }//End of spatial iteration loop


                        //Spatial and Temporal neighborhood processing - Right Eye
                        for(int j=1 ; j<= kernel_size ; j++){
                            for(int i=1 ; i <= kernel_size ; i++){

                                //Pixels to be processed in the spatial neighbourhood for both eyes - SPATIAL CORRELATION
                                int pixel_x = pos_x + i-6;
                                int pixel_y = pos_y + j-6;
                                double temporal_difference;

                                if(pixel_x >= 0 && pixel_y>= 0 && pixel_x < MAX_RES && pixel_y < MAX_RES){  //Checking for borders


                                //RIGHT EYE PROCESSING

                                //std::cout << "Event Processing Right Buffer..."<<std::endl;//Debug Code

                                event_history_right.timeStampsList_it = event_history_right.timeStampList[pixel_x][pixel_y].rbegin(); //Going from latest event pushed in the back

                                //NOTE : List size is always limited to the buffer size, took care of this in the event buffer code
                                if(!event_history_right.timeStampList[pixel_x][pixel_y].empty()){ //If the list is empty skip computing

                                    //std::cout << "Buffer Size Right Eye : " <<  event_history_right.timeStampList[pixel_x][pixel_y].size() << std::endl;//Debug Code

                                    for( int list_length = 1 ; list_length <= event_history_right.timeStampList[pixel_x][pixel_y].size() ; ){


                                        //NOTE : Timestamps in the list are encoded with polarity information
                                         temporal_difference = event_time - abs(*event_history_right.timeStampsList_it); //The first value is always zero
                                         //std::cout << "Temporal difference : " << temporal_difference <<std::endl; //Debug code

                                         //Here the neighborhood processing goes over 0 - 1 time scale
                                          if(temporal_difference > temporal_diff_step){
                                            //std::cout << "Skipping the list element..." << std::endl;//Debug Code
                                            ++list_length;
                                            continue;
                                         }


                                          //TODO Right Eye Each Phase processing
                                          double phase = 0;
                                          phase_vector_it = phase_vector.begin(); //Starting from the first
                                          for(int t=0; t < number_phases ; ){

                                              phase = *phase_vector_it; //Local Varialbe
                                              //std::cout << "Phase Value : "<<phase<<std::endl;//Debug Code
                                              //Call the spatio-temporal filtering function
                                              std::pair<double,double> conv_value = st_filters.filtering(pixel_x, pixel_y, theta, temporal_difference,phase);

                                              //TODO Check the effects of Polarity here! Without polarity its fine to get just the motion energy
                                              even_conv_right[t] = even_conv_right[t] +  conv_value.first;
                                              odd_conv_right[t] = odd_conv_right[t] +   conv_value.second;

                                              //std::cout << "Even Right : "<< even_conv_right[t] << std::endl; //Debug Code
                                              //std::cout << "Odd Right : "<< odd_conv_right[t] << std::endl; //Debug Code

                                              ++phase_vector_it;
                                              ++t;



                                          }



                                         ++event_history_right.timeStampsList_it; //Moving up the list
                                         ++list_length;



                                    }//End of temporal iteration loop

                                }
                                else {

                                    //std::cout << "Skipping empty list of Right Eye..." << std::endl; //Debug Code
                                    continue;
                                }
                          }


                           else{

                                 //std::cout<< "Pixels Out of Bordes...."<<std::endl; //Debug Code
                                 continue;
                               }

                           }
                        }//End of spatial iteration loop



                        //Energy Response of Quadrature pair, Even and Odd filters
                        for(int t=0; t < number_phases ; ){

                            final_convolution_even[t]= even_conv_left + even_conv_right[t];
                            final_convolution_odd[t]= odd_conv_left + odd_conv_right[t];
                            final_convolution[t] = (final_convolution_even[t]* final_convolution_even[t]) + (final_convolution_odd[t] * final_convolution_odd[t]);
                            //std::cout << "Final Convolution : " << final_convolution[t] << std::endl; //Debug Code

                            //TODO Chage the vector names
                            binocular_enery_theta.push_back(final_convolution[t]);
                            ++t;



                        }

                        //Writing values to direction specific file
                       /*if(dfile.is_open()){

                            dfile << pos_x <<" "<< pos_y <<" "<< event_time << " "<<final_convolution << "\n";

                        }
                        else {std::cout<<"Unable to open the text file to write!!!"<<std::endl;}

                        dfile.close();

                        if(fufile.is_open()){

                            fufile << pos_x <<" "<< pos_y <<" "<< event_time << " "<<final_convolution << "\n";

                        }
                        else {std::cout<<"Unable to open the text file to write!!!"<<std::endl;}

                        fufile.close();

                        //Pushing convolution values into a vector to store values for all directions
                        convolution.push_back(final_convolution);*/

                        //Normalizing Binocular Energy Response Values
                        double norm_sum =std::accumulate(binocular_enery_theta.begin(),binocular_enery_theta.end(),eta); //Check the effects of eta value
                        //std::cout << "Norm : " << norm_sum<<std::endl;//Debug Code

                        binocular_enery_theta_it = binocular_enery_theta.begin();
                        disparity_vector_it = disparity_vector.begin();
                        double phase_sum = 0; //Local variable
                        for(int t=0; t < number_phases ; ){

                            //Check the exact values accessed from the vectors
                            phase_sum = phase_sum + ( (*binocular_enery_theta_it * *disparity_vector_it )/norm_sum );

                            ++binocular_enery_theta_it;
                            ++disparity_vector_it;
                            ++t;

                        }

                        disparity_est_theta.push_back(phase_sum);

                        //std::cout << std::endl; //Debug Code

                        binocular_enery_theta.clear(); //Clearing the binocular energy vector of each phase


                        ++theta_vector_it;
                        ++theta_index;

                    } //End of spatio-temporal filtering

                    //std::cout << "Size of Disparity Estimation Vector : " << disparity_est_theta.size() << std::endl; //Debug Code

                    //Calling Disparity Estimation Function - Should be called with all the directions taking into account
                    disparity_estimation();

                    /*std::cout << "Event processing done..." << std::endl;//Debug Code
                    std::cout << "Size of convolution vector : " << convolution.size()<<std::endl;//Debug code

                    //Normalizing Energy Response Values
                    double norm_sum =std::accumulate(convolution.begin(),convolution.end(),eta); //Check the effects of eta value
                    std::cout << "Norm : " << norm_sum<<std::endl;//Debug Code
                    convolution_it = convolution.begin();
                    for(;convolution_it!=convolution.end();convolution_it++){

                        *convolution_it = *convolution_it/norm_sum;

                    }


                    //Finding the direection that has maximum value among all directions
                     auto max_value = std::max_element(convolution.begin(),convolution.end());
                     //std::cout << "Index of Maximum Value : "<< std::distance(std::begin(convolution), max_value) << std::endl;//Debug Code
                     //if(std::distance(std::begin(convolution), max_value)==0){

                         std::cout << std::distance(std::begin(convolution), max_value) << " " << *max_value << std::endl;

                     //}


                     //Clearing the Convolution Vector
                    convolution.clear();

                    ffile.open(flow_file.c_str(),ios::in | ios::app);*/
                    //motionHypothesis(pos_x,pos_y);

                //} //End of channel if, one event processing done



        }//End of event queue loop



   } //end of while loop


   std::cout<<"End of while loop"<<std::endl; //Debug code

    return 0;
}


/*void motionHypothesis(int &x,int &y){


    //std::cout << "Motion Estimation..."<<std::endl;//Debug Code
    //Code for single motion hypothesis generation


            //Velocity components set to zero
            double Ux = 0;
            double Uy = 0 ;

            theta_vector_it = theta_vector.begin();
            convolution_it = convolution.begin();
            theta_index = 0;
            //std::cout << "Convolution value after filtering : " << final_convolution << endl; //Debug Code

            for(; theta_vector_it != theta_vector.end() - 1 ; ){//Summing up motion estimation along all directons

                //std::cout << "Theta value : " << *theta_vector_it<< std::endl; //Debug Code

                Ux = Ux + ( *convolution_it  *  cos( *theta_vector_it ) )  ;
                Uy = Uy - ( *convolution_it  *  sin( *theta_vector_it ) ) ;

                //std::cout << "Velocity Components Ux : " << Ux << " Uy : " << Uy << std::endl; //Debug Code

                ++convolution_it;
                ++theta_vector_it;
                ++theta_index;

            }//End of theta loop

            //std::cout <<" "<<x<<" "<<y <<" " << Ux << " " << Uy << std::endl;//Debug Code

            if(ffile.is_open()){

                ffile << x<<" "<<y <<" " << Ux << " " << Uy << "\n";

            }
            else {std::cout<<"Unable to open the flow text file to write!!!"<<std::endl;}

            ffile.close();


            convolution.clear();
            //Calling quiver plot function
            //quiver_plot(x,y,Ux,Uy);



    }*/

/*void quiver_plot(int &x, int &y, double &ux, double &uy){


    double q_theta;
    cv::Point pt1,pt2;

    if(ux==0){ q_theta=M_PI /20; }
    else {

        q_theta = atan2(uy,ux);
    }

    pt1.x = y;
    pt1.y = x;

    pt2.x = y + uy;
    pt2.y = x + ux;

    std::cout << "Point 2 X: " <<pt2.x << " Y: " << pt2.y << std::endl;

    cv::line(eventImageMat,pt1,pt2,CV_RGB(0,255,0),1,8,0);
    cv::namedWindow("EVENT IMAGE",WINDOW_NORMAL);
    cv::imshow("EVENT IMAGE", eventImageMat);
    cv::waitKey(1);




}*/







