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
#define NUMPHASES 1 //Number of Phases

#include <cuda.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <cmath>
//#include <tuple>
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
#include "icubInterface.hpp"


using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::dev;
using namespace emorph;

using namespace std;
using namespace cv;

//CUDA Kernel for GPU Computation

//_global__ void convolutionKernerl(){

//}


//Global variables

int num_events=0;

//TODO Use one Single buffer with the channel information encoded in it
eventHistoryBuffer event_history;
int kernel_size;
double temporal_diff_step;


//Number of directions
int number_directions;
double theta_step;
double max_theta;
int theta_index;
std::vector<double> theta_vector;
std::vector<double>::iterator theta_vector_it;
double theta;

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


double even_conv_left;
double odd_conv_left;
//NOTE Check if this can be implmented in any other ways
double even_conv_right[NUMPHASES];
double odd_conv_right[NUMPHASES]; //Arrays of size = number of phases
double final_convolution_even[NUMPHASES];
double final_convolution_odd[NUMPHASES];
double final_convolution[NUMPHASES];
void disparity_estimation(int&,int&);
int vergence_control(double&);

std::vector<double> binocular_enery_theta;
std::vector<double>::iterator binocular_enery_theta_it;


//Instance of stFitlering class with filter parameters as arguments
int pos_x,pos_y;

//TODO Need to have different set of filters
stFilters st_filters(0.0625, 6.5, 0.0625, 5);

double eta; //Normalizing parameter for MT pooling

int initialize(){

    kernel_size = 31;
    temporal_diff_step = 1;


    //Number of directions - Changing the range from 0 to PI, NOTE
    max_theta =  M_PI;
    number_directions = 8;
    theta_step = 0.25;

    //Number of phases - In the range of - PI to PI
    max_phase = M_PI;
    number_phases = NUMPHASES; //Check this once again
    phase_step = (2*M_PI)/5; //Check this once again

    //eta = 0.0001; //TODO  check the value
    eta = 0;

    //Checking the Values of empty arrays
    /*for(int t=0;t<8;t++){

        std::cout<< "EVEN RIGHT VALUES : " << even_conv_right[t]<<std::endl;
        std::cout<< "ODD RIGHT VALUES : " << odd_conv_right[t]<<std::endl;

    }*/

    //Setting the theta vector based on the number of directios
    double value = 0;
    for(int p=1; p<=number_directions ;p++){ //Loop runs for number of directions

        //value = value + theta_step; //Line here in case of testing with single direction i.e theta
        theta_vector.push_back( value * M_PI);
        value = value + theta_step; //Line here in case all directions and with fixed theta step

    }

    //Setting the phase vector based on the number of phases
    double phase_gen[]={-1,-0.5,0,0.5,1};
    double phase_value=0; //Local Varialbe
    for(int t=0; t< number_phases; t++){

        //std::cout << "Phase Gen :"<< phase_gen[t] <<std::endl; //Debug Code
        phase_value = max_phase * phase_gen[t];

        phase_vector.push_back(phase_value);
        std::cout<<"Phase Values : "<< phase_value <<std::endl; //Debug Code

        disparity_vector.push_back(phase_value/st_filters.omega);
        std::cout<<"Disparity Values : "<< phase_value/st_filters.omega <<std::endl; //Debug Code

    }



}


void processing(){

    int current_event_channel;

    //Resetting filter convolution value for every event processing
    even_conv_left = 0;
    odd_conv_left = 0;

    //Right Eye convolution variables are stored in the array of size equal to number of phases
    std::fill_n(even_conv_right,number_phases,0); //Array initilization to zero
    std::fill_n(odd_conv_right,number_phases,0);



    //Spatial and Temporal neighborhood processing - Left Eye
    for(int j=1 ; j<= kernel_size ; j++){
        for(int i=1 ; i <= kernel_size ; i++){

            //Pixels to be processed in the spatial neighbourhood for both eyes - SPATIAL CORRELATION
            int pixel_x = pos_x + i-16;
            int pixel_y = pos_y + j-16;
            double temporal_difference;

            if(pixel_x >= 0 && pixel_y>= 0 && pixel_x < MAX_RES && pixel_y < MAX_RES){  //Checking for borders


                event_history.timeStampsList_it = event_history.timeStampList[pixel_x][pixel_y].rbegin(); //Going from latest event pushed in the back

                //NOTE : List size is always limited to the buffer size, took care of this in the event buffer code
                if(!event_history.timeStampList[pixel_x][pixel_y].empty()){ //If the list is empty skip computing

                    //std::cout << "Event History Buffer Size : " <<  event_history.timeStampList[pixel_x][pixel_y].size() << std::endl;//Debug Code

                    for( int list_length = 1 ; list_length <= event_history.timeStampList[pixel_x][pixel_y].size() ; ){


                        //NOTE : Timestamps in the list are encoded with polarity information
                         temporal_difference = event_time - abs(*event_history.timeStampsList_it); //The first value is always zero
                         //std::cout << "Temporal difference : " << temporal_difference <<std::endl; //Debug code

                         current_event_channel = (*event_history.timeStampsList_it)/(abs(*event_history.timeStampsList_it));

                         //Here the neighborhood processing goes over 0 - 1 time scale
                          if(temporal_difference > temporal_diff_step){
                            //std::cout << "Skipping the list element..." << std::endl;//Debug Code
                            ++list_length;
                            continue;
                         }

                         //NOTE: Now processing left and right channels based on the channel infomarion decoded from event history buffer
                         if(current_event_channel == -1){ //Left eye events


                             double phase=0.0; //Local Variable
                             //Call the spatio-temporal filtering function
                             std::pair<double,double> conv_value = st_filters.filtering(pixel_x, pixel_y, theta, temporal_difference,phase);

                             //TODO Check the effects of Polarity here! Without polarity its fine to get just the motion energy
                             even_conv_left = even_conv_left +  conv_value.first;
                             odd_conv_left = odd_conv_left +   conv_value.second;

                             //std::cout << "Left Eye EVEN and ODD : " << even_conv_left << " " << odd_conv_left << std::endl; //Debug Code

                         }

                         if(current_event_channel == 1){//Right eye events


                             double phase = 0;
                             phase_vector_it = phase_vector.begin(); //Starting from the first
                             for(int t=0; t < number_phases ; ){

                                 //std::cout << "Initial Even Right : "<< even_conv_right[t] << std::endl; //Debug Code
                                 //std::cout << "Initial Odd Right : "<< odd_conv_right[t] << std::endl; //Debug Code

                                 phase = *phase_vector_it; //Local Varialbe
                                 //std::cout << "Phase Value : "<<phase<<std::endl;//Debug Code
                                 //Call the spatio-temporal filtering function
                                 std::pair<double,double> conv_value = st_filters.filtering(pixel_x, pixel_y, theta, temporal_difference,phase);

                                 //TODO Check the effects of Polarity here! Without polarity its fine to get just the motion energy
                                 even_conv_right[t] = even_conv_right[t] +  conv_value.first;
                                 odd_conv_right[t] = odd_conv_right[t] +   conv_value.second;

                                 //std::cout << "Right Eye EVEN and ODD : " << even_conv_right[t] << " " << odd_conv_right[t] << std::endl; //Debug Code

                                 ++phase_vector_it;
                                 ++t;

                             }


                         }


                         ++event_history.timeStampsList_it; //Moving up the list
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

    //std::cout << "Left eye convolution values : " << even_conv_left << " " << odd_conv_left << std::endl; //Debug Code


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



   while(true){ //TODO wait till the input port receives data and run


    tempEvents = inputPort.read(true); //Read the incoming events into a temporary vBottle
    //std::cout << "received bottle" << std::endl;
    //continue;

    emorph::vQueue q; //Event queue
   	emorph::vQueue::iterator q_it;//Event queue iterator

    tempEvents->getAll(q); // 1 ms timeframe
    //std::cout << "Processing " << q.size()<< " events" <<std::endl;

    //for (q_it = q.begin(); q_it != q.end(); q_it++) //Processing each event
    while(!q.empty()) //Do the computation till the queue is empty
            {

                //std::cout << "Queue Size : " << q.size()<< " events" <<std::endl;
                q_it = q.begin();
                emorph::AddressEvent *aep = (*q_it)->getAs<emorph::AddressEvent>(); //Getting each event from the Queue
                q.pop_front();
                if(!aep) continue; //If AER is not received continue
                int channel = aep->getChannel(); //Processing events of both channel - LEFT and Right Eye


                //NOTE this is the new implementation with single event history buffer
                event_history.updateList(*aep); //Single event history buffer that stores both left and right eye information

                    //NOTE : Sensor X and Y are reversed
                    pos_x = aep->getY();
                    pos_y = aep->getX();

                    //std::cout<< "Processing Event at "<<"Channel: "<<channel<< " X : "<< pos_x <<" Y : "<< pos_y << std::endl; //Debug code

                    //Time stamp of the recent event
                    event_time = unwrap (aep->getStamp());

                    //converting to seconds
                    event_time = event_time / event_history.time_scale;


                    //Setting the filter spatial center values - Both Left and Right Filters are centered at the same spatial location - SPATIAL CORRELATION
                    st_filters.center_x = pos_x;
                    st_filters.center_y = pos_y;

                    //Processing for each direction i.e Theta
                    //std::cout << "Number of Directions : " << theta_vector.size()<<std::endl;//Debug Code
                    theta_vector_it = theta_vector.begin();
                    theta_index = 0;

                    for(; theta_vector_it != theta_vector.end() ; ){ //Processing for each theta

                        theta = *theta_vector_it;
                        //std::cout<<"Processing for Theta : " << theta<<std::endl;//Debug Code

                        processing(); //Calling the processing function

                        //Final Convolution values reset to zero
                        std::fill_n(final_convolution_even,number_phases,0);
                        std::fill_n(final_convolution_odd,number_phases,0);
                        std::fill_n(final_convolution,number_phases,0);

                        //Energy Response of Quadrature pair, Even and Odd filters - Simple Binocular Cell Response
                        disparity_vector_it = disparity_vector.begin();
                        for(int t=0; t < number_phases ; ){

                            //std::cout << "Computing Binocular Energy along each phase..." << std::endl;//Debug Code
                            //std::cout << "Left Eye EVEN and ODD : " << even_conv_left << " " << odd_conv_left << std::endl; //Debug Code
                            //std::cout << "Right Eye EVEN and ODD : " << even_conv_right[t] << " " << odd_conv_right[t] << std::endl; //Debug Code

                            final_convolution_even[t]= even_conv_left + even_conv_right[t];
                            final_convolution_odd[t]= odd_conv_left + odd_conv_right[t];

                            //std::cout << "Final Convolution EVEN and ODD : " << final_convolution_even[t] << " " << final_convolution_odd[t] << std::endl; //Debug Code

                            final_convolution[t] = (final_convolution_even[t]*final_convolution_even[t]) + (final_convolution_odd[t] * final_convolution_odd[t]); //Binocular Energy Computation
                            std::cout << theta_index << " " << *disparity_vector_it << " " << final_convolution[t] << std::endl ; //Debug Code

                            binocular_enery_theta.push_back(final_convolution[t]);
                            disparity_vector_it++;
                            ++t;


                        }


                        //Normalizing Binocular Energy Response Values
                        //double energy_sum =std::accumulate(binocular_enery_theta.begin(),binocular_enery_theta.end(),eta); //Check the effects of eta value
                        //std::cout << "Energy Sum : " << energy_sum<<std::endl;//Debug Code

                        binocular_enery_theta_it = binocular_enery_theta.begin(); //The size should be 8 for single phase
                        disparity_vector_it = disparity_vector.begin();
                        double phase_sum = 0; //Local variable
                        double energy_sum = 0;
                        for(int t=0; t < number_phases ; ){

                            //std::cout << "Disparity value : " << *disparity_vector_it << std::endl; //Debug Code

                            energy_sum = energy_sum + *binocular_enery_theta_it;

                            //std::cout << "Binocular Energy : " << *binocular_enery_theta_it << std::endl; //Debuc Code
                            phase_sum = phase_sum + (*binocular_enery_theta_it * *disparity_vector_it );

                            ++binocular_enery_theta_it;
                            ++disparity_vector_it;
                            ++t;

                        }

                        //std::cout <<"Energy sum along all phases : " << phase_sum << std::endl;//Debug Code
                        //phase_sum = phase_sum/energy_sum; //Normalizing
                        //std::cout << phase_sum << std:: endl;
                        //std::cout << "Disparity Estimation Theta along all phase : " << phase_sum << std::endl; //Debug Code

                        disparity_est_theta.push_back(energy_sum); //This is Complex  Binocular Cell response along all phases at each direction

                        //std::cout << std::endl; //Debug Code

                        //binocular_enery_theta.clear(); //Clearing the binocular energy vector of each theta
                        ++theta_vector_it;
                        ++theta_index;

                    } //End of spatio-temporal filtering

                    //std::cout << "Size of Disparity Estimation Vector : " << disparity_est_theta.size() << std::endl; //Debug Code

                    //std::cout << "Number of events processed : " << num_events << std::endl; //Debug Code
                    //Calling Disparity Estimation Function - Should be called with all the directions taking into account
                    disparity_estimation(pos_x,pos_y);

                    
        }//End of event queue loop



   } //end of while loop


   std::cout<<"End of while loop"<<std::endl; //Debug code

    return 0;
}

//Final Disparity Estimation
void disparity_estimation(int &x,int &y){


    //std::cout << "Motion Estimation..."<<std::endl;//Debug Code
    //Code for single motion hypothesis generation


            //Disparity components set to zero
            double Dx = 0;
            double Dy = 0 ;

            theta_vector_it = theta_vector.begin();
            disparity_est_theta_it = disparity_est_theta.begin();
            theta_index = 0;
            //std::cout << "Convolution value after filtering : " << final_convolution << endl; //Debug Code


            for(; theta_vector_it != theta_vector.end() - 1 ; ){//Summing up motion estimation along all directons

                //std::cout << "Theta value : " << *theta_vector_it<< std::endl; //Debug Code

                Dx = Dx + ( *disparity_est_theta_it   *  cos( *theta_vector_it ) )  ;
                Dy = Dy - ( *disparity_est_theta_it   *  sin( *theta_vector_it ) ) ;

                //std::cout << "Velocity Components Ux : " << Ux << " Uy : " << Uy << std::endl; //Debug Code

                ++disparity_est_theta_it ;
                ++theta_vector_it;
                ++theta_index;

            }//End of theta loop

            std::cout <<" "<<x<<" "<<y <<" " << Dx << " " << Dy << std::endl;//Debug Code

            //Clearing the disparity estimation vector
            disparity_est_theta.clear();




    }
