#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>

#include "Agent.h"
#include "logging.h"


__host__ __device__ int ind(int row, int col){
	return row * (row + 1) / 2 + col;
}

__host__ __device__ int compi(int x){
	if(x % 2 == 0) return x + 1;
	else return x - 1;
}

__host__ __device__ bool get_asymmetry_dir(bool *dir, int row, int col){
	return dir[ind(compi(col), compi(row))];
}

//helper function
/*
This function does the conjunction for two lists
Input: two bool lists
Output: None
*/
__global__ void conjunction_kernel(bool *b1, bool *b2, int size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < size){
		b1[index] = b1[index] && b2[index];
	}
}

/*
This function computes the difference of two lists
Input: two bool lists (same length)
Output: None
*/
__global__ void subtraction_kernel(bool *b1, bool *b2, int size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < size){
		b1[index] = b1[index] && !b2[index];
	}
}

/*
This function does the disjunction for two lists
Input: two bool lists
Output: None
*/
__global__ void disjunction_kernel(bool *b1, bool *b2, int size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < size){
		b1[index] = b1[index] || b2[index];
	}
}

/*
This function does compi, conjunction together
Input: two boolean lists
Output: None
*/
__global__ void negate_conjunction_star_kernel(bool *b1, bool *b2, int size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < size){
		if(index%2 == 0){
			b1[index] = b1[index] && !b2[index+1];
		}
		else{
			b1[index] = b1[index] && !b2[index-1];
		}
	}
}

__global__ void conjunction_star_kernel(bool *b1, bool *b2, int size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < size){
		if(index%2 == 0){
			b1[index] = b1[index] && b2[index+1];
		}
		else{
			b1[index] = b1[index] && b2[index-1];
		}
	}
}

/* Deprecated in this version of code
This function convert int value to bool, 1->true, other->false
Input: int list and distination bool list
Output: None
*/
__global__ void int2bool_kernel(bool *b, int *i, int size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < size){
		if(i[index] == 1) b[index] = true;
		else b[index] = false;
	}
}

/* Deprecated in this version of code
This function convert bool to int value, true->1, false->0
Input: bool list and distination int list
Output: None
*/
__global__ void bool2int_kernel(int *i, bool *b, int size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < size){
		if(b[index]) i[index] = 1;
		else i[index] = 0;
	}
}

__global__ void up2down(bool *b1, bool *b2, int size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < size){
		b2[index] = !b1[compi(index)];
	}
}
//---------------------helper function------------------------


//---------------------non-worker solution------------------------

__device__ double lower_threshold(double q, double T){
	return q * T;
}

__device__ double raise_threshold(double q, double T){
	return ( 1.0 / q ) * T;
}

__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

__host__ __device__ thrust::default_random_engine random_engine(
        int iter, int index = 0, int depth = 0) {
    return thrust::default_random_engine(utilhash((index + 1) * iter) ^ utilhash(depth));
}

__device__ float getRandomNum(int iter,int index,int depth){
	thrust::uniform_real_distribution<float> u01(0, 1);
	thrust::default_random_engine rng = random_engine(iter, index, depth);
	return u01(rng);
}

/*
This function does implies on GPU, using non-worker solution
Input: row, col info, measurable size, weight matrix and threshold
Output: bool value(mathematical info please inquiry Kotomasha)
*/
__device__ bool implies_GPU(int row, int col, double *weights, double total, double threshold){//implies
	double rc = weights[ind(row, col)];
	double r_c = weights[ind(compi(row), col)];
	double rc_ = weights[ind(row, compi(col))];
	double r_c_ = weights[ind(compi(row), compi(col))];
	double epsilon = total * threshold;
	double m = min(epsilon,min(rc, min(r_c, r_c_)));
	return rc_ < m;
}

/*
This function does equivalent on GPU, using non-worker solution
Input: row, col info, measurable size, weight matrix and threshold
Output: bool value(mathematical info please inquiry Kotomasha)
*/
__device__ bool equivalent_GPU(int row, int col, double *weights){//equivalent
	double rc = weights[ind(row, col)];
	double r_c = weights[ind(compi(row), col)];
	double rc_ = weights[ind(row, compi(col))];
	double r_c_ = weights[ind(compi(row), compi(col))];
	//double epsilon = (rc + r_c + rc_ + r_c_) * threshold;
	return rc_ == 0 && r_c == 0 && rc > 0 && r_c_ > 0;
}

/*
This function does orient_square on GPU, using non-worker solution
Input: direction, weight, threshold matrix, xy location, and measurable size
Output: None
*/
__device__ void orient_square_GPU(bool *dir, double *weights, double *thresholds, double total, double q, int x, int y, int width){//orient_square
	bool old_y_x = dir[ind(y, x)];
	bool old_cy_x = dir[ind(compi(y), x)];
	bool old_y_cx = dir[ind(y, compi(x))];
	bool old_cy_cx = dir[ind(compi(y), compi(x))];
	double threshold = thresholds[ind(y, x)];
	dir[ind(y, x)] = implies_GPU(y, x, weights, total, threshold) || equivalent_GPU(y, x, weights);
	dir[ind(compi(y), x)] = implies_GPU(compi(y), x, weights, total, threshold) || equivalent_GPU(compi(y), x, weights);
	dir[ind(y, compi(x))] = implies_GPU(y, compi(x), weights, total, threshold) || equivalent_GPU(y, compi(x), weights);
	dir[ind(compi(y), compi(x))] = implies_GPU(compi(y), compi(x), weights, total, threshold) || equivalent_GPU(compi(y), compi(x), weights);

	bool changed = false;
	//SIQI:if(!old_y_x && dir[ind(y, x)] && threshold >= 1 - q){
	if(dir[ind(y, x)] && threshold >= 1 - q){
		changed = true;
		thresholds[ind(y, x)] = lower_threshold(q, threshold);
	}
	//SIQI:if(!changed && !old_cy_x && dir[ind(compi(y), x)] && threshold >= 1 - q){
	if(!changed && dir[ind(compi(y), x)] && threshold >= 1 - q){
		changed = true;
		thresholds[ind(y, x)] = lower_threshold(q, threshold);
	}
	//SIQI:if(!changed && !old_y_cx && dir[ind(y, compi(x))] && threshold >= 1 - q){
	if(!changed && dir[ind(y, compi(x))] && threshold >= 1 - q){
		changed = true;
		thresholds[ind(y, x)] = lower_threshold(q, threshold);
	}
	//SIQI:if(!changed && !old_cy_cx && dir[ind(compi(y), compi(x))] && threshold >= 1-q){
	if(!changed && dir[ind(compi(y), compi(x))] && threshold >= 1 - q){
		changed = true;
		thresholds[ind(y, x)] = lower_threshold(q, threshold);
	}

	if(!changed && old_y_x && !dir[ind(y, x)]){
		changed = true;
		thresholds[ind(y, x)] = raise_threshold(q, threshold);
	}
	if(!changed && old_cy_x && !dir[ind(compi(y), x)]){
		changed = true;
		thresholds[ind(y, x)] = raise_threshold(q, threshold);
	}
	if(!changed && old_y_cx && !dir[ind(y, compi(x))]){
		changed = true;
		thresholds[ind(y, x)] = raise_threshold(q, threshold);
	}
	if(!changed && old_cy_cx && !dir[ind(compi(y), compi(x))]){
		changed = true;
		thresholds[ind(y, x)] = raise_threshold(q, threshold);
	}
}

/*
This function is update weights for discounted agent, using non-worker solution
Input: weight matrix, observe bool value from python side and measurable size
Output: None
*/
//SIQI: I've removed "using_agent" and put "activity" instead and put TWO variants (one of them commented out) of the update that I want to stay here.
__global__ void update_weights_kernel_discounted(double *weights, bool *observe, int measurable_size, double q, double phi, bool activity){
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	if(indexX <= indexY && indexY < measurable_size){
	    /*
	    // INDEPENDENT UPDATES FOR PLUS AND MINUS
	    if(activity) {
	        weights[ind(indexY,indexX)]=q * weights[ind(indexY,indexX)] + (1-q) * observe[indexX] * observe[indexY] * phi;
	    }
	    */
	    
	    // UPDATES ADD UP TO A UNIFIED SNAPSHOT UPDATE
	    weights[ind(indexY, indexX)] = weights[ind(indexY, indexX)] * q + (1 - q) * observe[indexX] * observe[indexY] * activity * phi;
	}
}

__global__ void get_measurable_kernel(double *weights, double *measurable, double *measurable_old, int measurable_size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < measurable_size){
		int idx = ind(index, index);
		measurable_old[index] = measurable[index];
		measurable[index] = weights[idx];
	}
}

__global__ void calculate_target_kernel(double *measurable, bool *target, int sensor_size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < sensor_size){
		if(measurable[2 * index] - measurable[2 * index + 1] > 1e-12){
			target[2 * index] = true;
			target[2 * index + 1] = false;
		}
		else if(measurable[2 * index] - measurable[2 * index + 1] < 1e-12){
			target[2 * index] = false;
			target[2 * index + 1] = true;
		}
		else{
			target[2 * index] = false;
			target[2 * index + 1] = false;
		}
		/*
		else{
			thrust::default_random_engine rng = random_engine((int)(100*diagonal[2*index]), (int)(100*diagonal[2*index+1]), 0);
			thrust::uniform_real_distribution<float> u01(0, 1);
			/*if(u01(rng)>0.5){
				target[2 * index] = true;
				target[2 * index + 1] = false;
			}
			else{
				target[2 * index] = false;
				target[2 * index + 1] = true;
			}
			target[2 * index] = false;
			target[2 * index + 1] = false;
		}
		*/
	}
}

/*
This function is orient all on GPU, using non-worker solution
Input: direction, weight, threshold matrix, and measurable size
Output: None
*/
__global__ void orient_all_kernel(bool *dir, double *weights, double *thresholds, double total, double q, int measurable_size){
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	if(indexY < measurable_size / 2 && indexX <= indexY){
		orient_square_GPU(dir, weights, thresholds, total, q, indexX * 2, indexY * 2, measurable_size);
	}
}

/*
This function is dfs on GPU, using non-worker solution
This function use shared memory and thus has only one GPU block, the default threshold number is 1024
Input: bool list of data to be dfsed, direction matrix and measurable size
Output: None
*/
__global__ void multiply_kernel(bool *x, bool *dir, double *thresholds, bool is_stable, double lowest, int size){//dfs
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	extern __shared__ bool shared[];
	bool *xs = &shared[0];
	bool *ys = &shared[size];
	__shared__ bool flag[1];
	int j = index;
	// matrix multiplication variable
	while(j < size) {
		xs[j] = x[j];
		ys[j] = x[j];
		j += 1024;
	}
	flag[0] = true;
	__syncthreads();
	while(flag[0]){
		flag[0] = false;
		__syncthreads();
		j = index;
		while(j < size){
			if(xs[j] == 1){
				j += 1024;
				continue;
			}
			for(int i = 0; i < size; ++i){
				if((i >= j && dir[ind(i,j)]) & xs[i] == 1 && (!is_stable || thresholds[ind(i - i % 2, j - j % 2)] < lowest)){
					ys[j] = 1;
					flag[0] = true;
					break;
				}
				else if((i < j && get_asymmetry_dir(dir, i, j)) & xs[i] == 1 && (!is_stable || thresholds[ind(i - i % 2, j - j % 2)] < lowest)){
					ys[j] = 1;
					flag[0] = true;
					break;
				}
			}
			j += 1024;
		}
		__syncthreads();
		j = index;
		while(j < size){
			xs[j] = ys[j];
			j += 1024;
		}
		__syncthreads();
	}
	j = index;
	while(j < size){
		x[j] = ys[j];
		j += 1024;
	}
}
//---------------------non-worker solution------------------------

//mask=Signal([(ind in actions_list) for ind in xrange(self._SIZE)])
/*
This function is the GPU version of python function mask, it is designed to get mask signal
Deprecated in new version
Input: destination mask address, action list and mask size
Output: None
*/
__global__ void mask_kernel(bool *mask_amper, bool *mask, bool *current, int sensor_size){
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	if(indexX <= indexY && indexY < sensor_size && (mask_amper[2 * ind(indexY, indexX)] || mask_amper[2 * ind(indexY, indexX) + 1])){//need trick in the future to improve performance
		if(mask[2 * indexY]){//means still need check
			if(mask_amper[2 * ind(indexY, indexX)]){//check pure position
				if(!current[2 * indexX]) mask[2 * indexY] = false;
			}
			else{//check '*' position
				if(!current[2 * indexX + 1]) mask[2 * indexY] = false;
			}
		}
	}
}

__global__ void check_mask(bool *mask, int sensor_size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < sensor_size){
		if(mask[2 * index]) mask[2 * index + 1] = false;
	}
}

__global__ void delta_weight_sum_kernel(double *measurable, bool *signal, float *result, int sensor_size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < sensor_size){
		atomicAdd(result, signal[2 * index] * (measurable[2 * index] - measurable[2 * index + 1]) + signal[2 * index + 1] * (measurable[2 * index + 1] - measurable[2 * index]));
	}
}

/*
This function is an independent up function on GPU
It only use signal to do dfs, result is stored in Gsignal after using the function
Input: signal to be dfsed
Output: None
*/
void Agent::up_GPU(vector<bool> signal, bool is_stable){
	//logging_info->append_log(logging::UP, "[UP]:\n");
	//logging_info->add_indent();
	for(int i = 0; i < measurable_size; ++i) Gsignal[i] = signal[i];
	cudaMemcpy(dev_signal, Gsignal, measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
	//logging_info->append_log(logging::UP, "Separate Up Process is Invoked by Agent: "+name+"\n");
	//logging_info->append_log(logging::UP, "Size of Signal: "+to_string(whole_size)+"\n");
	
	multiply_kernel<<<1, 1024, 2 * measurable_size * sizeof(bool)>>>(dev_signal, dev_dir, dev_thresholds, is_stable, 1-q, measurable_size);

	cudaMemcpy(Gup, dev_signal, measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	
	up2down<<<(measurable_size + 255) / 256, 256>>>(dev_signal, dev_load, measurable_size);

	cudaMemcpy(Gdown, dev_load, measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);

	//logging_info->append_process(logging::UP, logging::PROCESS);
	//logging_info->reduce_indent();
}

void Agent::gen_mask(){
	//cudaMemset(dev_mask, true, measurable_size * sizeof(bool));
	cudaMemset(dev_mask, false, 2 * this->base_sensor_size * sizeof(bool));
	cudaMemset(dev_mask + 2 * this->base_sensor_size, true, (measurable_size - 2 * this->base_sensor_size) * sizeof(bool));

	dim3 dimGrid((sensor_size + 15) / 16,(sensor_size + 15) / 16);
	dim3 dimBlock(16, 16);

	mask_kernel<<<dimGrid, dimBlock>>>(dev_mask_amper, dev_mask, dev_current, this->sensor_size);
	check_mask<<<(this->sensor_size + 255) / 256, 256>>>(dev_mask, this->sensor_size);
}

/*
This function do propagate on GPU
//before invoke this function make sure dev_load and dev_signal have correct data
//the computed data will be in dev_load
Result is stored in Gload
Ask Kotomasha for mathematic questions
Input: signal and load
Output: None
*/
void Agent::propagate_GPU(vector<bool> signal, vector<bool> load, bool t){//propagate
	//logging_info->record_start();

	if(!signal.empty()){
		for(int i = 0; i < signal.size(); ++i){
			Gsignal[i] = signal[i];
			Gload[i] = load[i];
		}
		cudaMemcpy(dev_load, Gload, measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_signal, Gsignal, measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
	}

	//don't forget to add logging system
	multiply_kernel<<<1, 1024, 2 * measurable_size * sizeof(bool)>>>(dev_load, dev_dir, dev_thresholds, false, 0, measurable_size);
	multiply_kernel<<<1, 1024, 2 * measurable_size * sizeof(bool)>>>(dev_signal, dev_dir, dev_thresholds, false, 0, measurable_size);

	// standard operations
	disjunction_kernel<<<(measurable_size + 255) / 256, 256>>>(dev_load, dev_signal, measurable_size);
	negate_conjunction_star_kernel<<<(measurable_size + 255) / 256, 256>>>(dev_load, dev_signal, measurable_size);
	
	cudaMemcpy(Gload, dev_load, measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
}

/*
This function set signal for simulation in each frame
Input: observe value from python side
Output: None
*/
void Agent::setSignal(vector<bool> observe){//this is where data comes in in every frame
	for(int i = 0; i < observe.size(); ++i){
		Gobserve[i] = observe[i];
	}
	cudaMemcpy(dev_observe, Gobserve, measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
}

void Agent::calculate_total(bool active){}

void Agent::calculate_target(){}

/*
This function is update_weights based function for all types of agents
Input: None
Output: None
*/
void Agent::update_weights(bool active){}

void Agent::orient_all(){}

void Agent::update_thresholds(){}

//SIQI: Same here, I've removed "using agent". Please keep it that way.
void Agent_Discounted::calculate_total(bool activity){
	get_measurable_kernel<<<(measurable_size + 255) / 256, 256>>>(dev_weights, dev_measurable, dev_measurable_old, measurable_size);
	//thrust::device_ptr<double> cptr = thrust::device_pointer_cast(dev_diagonal);
	//total = thrust::reduce(cptr, cptr + whole_size) / measurable_size;
	last_total = total;
	// INDEPENDENT UPDATE FOR PLUS AND MINUS SNAPSHOTS:
	/*
	if(activity) {
	    total = q * total + (1-q) * phi
	}

	*/
	// PLUS AND MINUS UPDATES ADD UP TO A 'FULL' SNAPHOT:
	total = q * total  +  (1-q) * phi * activity;
}

void Agent_Discounted::calculate_target(){
	calculate_target_kernel<<<(sensor_size + 255) / 256, 256>>>(dev_measurable, dev_target, sensor_size);
	cudaMemcpy(Gtarget, dev_target, measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
}

/*
This function is update weights on GPU for discounted agent
It uses the update_weights_kernel_discounted, see that for detail
Input: None
Output: None
*/
//SIQI: here, too, "active" and "using agent" have been removed, and replaced with the Python-supplied "activity". We'll keep things this way until it is time to move the full Agent class to the cpp side.
void Agent_Discounted::update_weights(bool activity){
	//logging_info->record_start();
	dim3 dimGrid2((measurable_size + 15) / 16, (measurable_size + 15) / 16);
	dim3 dimBlock2(16, 16);
	update_weights_kernel_discounted<<<dimGrid2, dimBlock2>>>(dev_weights, dev_observe, measurable_size, q, phi, activity);
	//logging_info->record_stop(logging::UPDATE_WEIGHT);
}



/*
This function is orient all on GPU
See orient_all_kernel for more detail
Input: None
Output: None
*/
void Agent_Discounted::orient_all(){
	//logging_info->record_start();
	dim3 dimGrid1((measurable_size / 2 + 15) / 16,(measurable_size / 2 + 15) / 16);
	dim3 dimBlock1(16, 16);
	orient_all_kernel<<<dimGrid1, dimBlock1>>>(dev_dir, dev_weights, dev_thresholds, total, q, measurable_size);
}

/*
This function start record cuda events, only using_log is true will record event
Input: None
Output: None
*/
void logging::record_start(){
	if(!using_log) return;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
}

/*
This function stop record cuda events, only using_log is true will record event
The time used is calculated and added to the corresponding statistics
Input: None
Output: None
*/
void logging::record_stop(int LOG_TYPE){
	if(!using_log) return;
	float dt = 0;
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&dt, start, stop);
	switch(LOG_TYPE){
	case logging::UPDATE_WEIGHT:
		n_update_weight++;
		t_update_weight += dt;
		break;
	case logging::ORIENT_ALL:
		n_orient_all++;
		t_orient_all += dt;
		break;
	case logging::PROPAGATION:
		n_propagation++;
		t_propagation += dt;
		break;
	}
}

/*
This function update state on GPU, it is the main function for simulation on C++
It contains three main parts: update weight, orient all, propagation, result will be stored in Gload(propagate_GPU)
Input: mode to use
Output: None
*/
void Agent::update_state_GPU(bool activity){//true for decide	
        // udpate the snapshot weights and total count:
     	update_weights(activity);
	calculate_total(activity);

	// compute the derived orientation matrix and update the thresholds:
	orient_all();

	// compute the target state:
	calculate_target();

	cudaMemcpy(dev_signal, dev_observe, measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	cudaMemset(dev_load, false, measurable_size * sizeof(bool));
	vector<bool> tmp_signal, tmp_load;
	propagate_GPU(tmp_signal, tmp_load, false);

	cudaMemcpy(Gcurrent, dev_load, measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(dev_current, dev_load, measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	//cudaMemcpy(Gdir, dev_dir, whole_size * whole_size * sizeof(bool), cudaMemcpyDeviceToHost);

	t++;
}

int Agent::distance(bool *signal1, bool *signal2){
	cudaMemcpy(dev_signal, signal1, measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	cudaMemset(dev_load, false, measurable_size * sizeof(bool));
	vector<bool> tmp_signal, tmp_load;
	propagate_GPU(tmp_signal, tmp_load,false);
	cudaMemcpy(dev_signal1, dev_load, measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);

	cudaMemcpy(dev_signal, signal2, measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	cudaMemset(dev_load, false, measurable_size * sizeof(bool));
	propagate_GPU(tmp_signal, tmp_load, true);
	cudaMemcpy(dev_signal2, dev_load, measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);

	conjunction_star_kernel<<<(measurable_size + 255) / 256, 256>>>(dev_signal1, dev_signal2, measurable_size);
	cudaMemcpy(Gsignal, dev_signal1, measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	int sum = 0;
	for(int i = 0; i < measurable_size; ++i) sum += Gsignal[i];
	
	return sum;
}

float Agent::distance_big(bool *signal1, bool *signal2){
	float result = 0;
	float *dev_result;
	cudaMalloc(&dev_result, sizeof(float));
	cudaMemset(dev_result, 0, sizeof(float));

    cudaMemcpy(dev_signal, signal1, measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	cudaMemset(dev_load, false, measurable_size * sizeof(bool));
	vector<bool> tmp_signal, tmp_load;
	propagate_GPU(tmp_signal, tmp_load,false);
	cudaMemcpy(dev_signal1, dev_load, measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);

	cudaMemcpy(dev_signal, signal2, measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	cudaMemset(dev_load, false, measurable_size * sizeof(bool));
	propagate_GPU(tmp_signal, tmp_load, true);
	cudaMemcpy(dev_signal2, dev_load, measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);

	subtraction_kernel<<<(measurable_size + 255) / 256, 256>>>(dev_signal1, dev_signal2, measurable_size);

	/*
	This generates an all-ones weight vector
	*/
	double *allonesvec;
	cudaMalloc(&allonesvec, measurable_size * sizeof(double));
	cudaMemset(allonesvec,1.0,measurable_size * sizeof(double));

	/*
	Raw bit-count option:
	*/
	delta_weight_sum_kernel<<<(sensor_size + 255) / 256, 256>>>(allonesvec, dev_signal1, dev_result, sensor_size);

	/*
	Weighted bit-count option:
	*/	
	//delta_weight_sum_kernel<<<(sensor_size + 255) / 256, 256>>>(dev_measurable, dev_signal1, dev_result, sensor_size);

	cudaMemcpy(&result, dev_result, sizeof(double), cudaMemcpyDeviceToHost);
	/*
	To be removed if allones vector is not needed:
	*/
	cudaFree(allonesvec);

	cudaFree(dev_result);

	return result;
}

/*
This function is halucinate on GPU, it use several propagate_GPU
It first get the mask to use and then use propagate
The output will be stored in Gload(propagate_GPU)
Input: action list to be halucinated
Output: None
*/
void Agent::halucinate_GPU(){
	gen_mask();

	cudaMemcpy(dev_signal, dev_mask, measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	//cudaMemcpy(dev_load, dev_current, whole_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	cudaMemset(dev_load, false, measurable_size * sizeof(bool));
	vector<bool> tmp_signal, tmp_load;
	propagate_GPU(tmp_signal, tmp_load, false);
	cudaMemcpy(Gload, dev_load, measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(Gprediction, dev_load, measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
}

/*
This function free data together
Deprecated in new version
Input: None
Output: None
*/
void Agent::free_other_parameters(){//free data in case of memory leak
	delete[] Gobserve;
	delete[] Gsignal;
	delete[] Gload;
	delete[] Gmask;
	delete[] Gcurrent;
	delete[] Gtarget;
	delete[] GMeasurable;
	delete[] GMeasurable_old;
	delete[] Gprediction;
	delete[] Gup;
	delete[] Gdown;

	cudaFree(dev_mask);
	cudaFree(dev_current);
	cudaFree(dev_target);

	cudaFree(dev_observe);
	cudaFree(dev_signal);
	cudaFree(dev_load);
	
	cudaFree(dev_signal1);
	cudaFree(dev_signal2);
	cudaFree(dev_measurable);
	cudaFree(dev_measurable_old);
}

/*
This function generate direction matrix
It will first delete the old Matrix
And then generate new one
Log info is saved
Input: log type, indicating where this function is invoked
Output: None
*/
void Agent::gen_direction(int P_TYPE){
	delete[] Gdir;

	Gdir = new bool[array_size_max];
	//logging_info->append_log(P_TYPE, "Direction Matrix Generated, Size of Matrix: "
		//+ to_string(measurable_size) + "*"+to_string(measurable_size) + "\n");
	//logging::add_CPU_MEM(measurable_size * measurable_size * sizeof(bool));

	cudaFree(dev_dir);

	cudaMalloc(&dev_dir, array_size_max * sizeof(bool));
	//logging_info->append_log(P_TYPE, "GPU Memory Malloced for Direction Matrix: "
		//+ to_string(measurable_size * measurable_size * sizeof(bool)) + " Bytes(" + to_string
		//(measurable_size * measurable_size * sizeof(bool) / (1024.0 * 1024)) + "M)\n");
	//logging::add_GPU_MEM(measurable_size * measurable_size * sizeof(bool));
}

/*
This function generate weight matrix
It will first delete the old Matrix
And then generate new one
Log info is saved
Input: log type, indicating where this function is invoked
Output: None
*/
void Agent::gen_weight(int P_TYPE){
	delete[] Gweights;

	Gweights = new double[array_size_max];
	//logging_info->append_log(P_TYPE, "Weight Matrix Generated, Size of Matrix: "
		//+ to_string(measurable_size) + "*" + to_string(measurable_size) + "\n");
	//logging::add_CPU_MEM(measurable_size * measurable_size * sizeof(double));
	
	cudaFree(dev_weights);

	cudaMalloc(&dev_weights, array_size_max * sizeof(double));
	//logging_info->append_log(P_TYPE, "GPU Memory Malloced for Weight Matrix: "
		//+ to_string(measurable_size * measurable_size * sizeof(double)) + " Bytes(" + to_string
		//(measurable_size * measurable_size * sizeof(double) / (1024.0 * 1024)) + "M)\n");
	//logging::add_GPU_MEM(measurable_size * measurable_size * sizeof(double));
}

/*
This function generate thresholds matrix
It will first delete the old Matrix
And then generate new one
Log info is saved
Input: log type, indicating where this function is invoked
Output: None
*/
void Agent::gen_thresholds(int P_TYPE){
	delete[] Gthresholds;

	Gthresholds = new double[array_size_max];
	//logging_info->append_log(P_TYPE, "Thresholds Matrix Generated, Size of Matrix: "
		//+ to_string(measurable_size) + "*" + to_string(measurable_size) + "\n");
	//logging::add_GPU_MEM(measurable_size * measurable_size * sizeof(double));
	
	cudaFree(dev_thresholds);

	cudaMalloc(&dev_thresholds, array_size_max * sizeof(double));
	//logging_info->append_log(P_TYPE, "GPU Memory Malloced for Thresholds Matrix: "
		//+ to_string(measurable_size * measurable_size * sizeof(double)) + " Bytes(" + to_string
		//(measurable_size * measurable_size * sizeof(double) / (1024.0 * 1024)) + "M)\n");
	//logging::add_GPU_MEM(measurable_size * measurable_size * sizeof(double));
}

void Agent::gen_mask_amper(int P_TYPE){
	delete[] GMask_amper;

	GMask_amper = new bool[mask_amper_size_max];

	cudaFree(dev_mask_amper);

	cudaMalloc(&dev_mask_amper, mask_amper_size_max * sizeof(bool));
}

/*
This function copy direction matrix from CPU to GPU.
If no data is provided, use default method, otherwise use the data value
Input: data to copy to GPU, log type, indicating where this function is invoked
Output: None
*/
void Agent::init_direction(int P_TYPE){
	int x = 0, y = 0;
	for(int i = 0; i < array_size; ++i){
		Gdir[i] = (x == y);
		x++;
		if(x > y){
			y++;
			x = 0;
		}
	}
	cudaMemcpy(dev_dir, Gdir, array_size * sizeof(bool), cudaMemcpyHostToDevice);
	//logging_info->append_log(P_TYPE, "Direction Matrix Copied to GPU\n");
}

/*
This function copy weight matrix from CPU to GPU.
If no data is provided, use default method(=0.0), otherwise use the data value
Input: data to copy to GPU, log type, indicating where this function is invoked
Output: None
*/
void Agent::init_weight(int P_TYPE){
	for(int i = 0; i < array_size; ++i){
		Gweights[i] = 0.0;
	}
	cudaMemcpy(dev_weights, Gweights, array_size * sizeof(double), cudaMemcpyHostToDevice);
	//logging_info->append_log(P_TYPE, "Weight Matrix Copied to GPU\n");
}

/*
This function copy thresholds matrix from CPU to GPU.
If no data is provided, use default method(=threshold), otherwise use the data value
Input: data to copy to GPU, log type, indicating where this function is invoked
Output: None
*/
void Agent::init_thresholds(int P_TYPE){
	for(int i = 0; i < array_size; ++i){
		Gdir[i] = threshold;
	}
	cudaMemcpy(dev_thresholds, Gthresholds, array_size * sizeof(double), cudaMemcpyHostToDevice);
	//logging_info->append_log(P_TYPE, "Thresholds Matrix Copied to GPU\n");
}

void Agent::init_mask_amper(int P_TYPE){
	cudaMemset(dev_mask_amper, false, mask_amper_size_max * sizeof(bool));
}

void Agent::init_other_parameter(int LOG_TYPE){
	cudaMemset(&dev_observe, false, measurable_size_max * sizeof(bool));
	cudaMemset(&dev_signal, false, measurable_size_max * sizeof(bool));
	cudaMemset(&dev_load, false, measurable_size_max * sizeof(bool));
	cudaMemset(&dev_mask, false, measurable_size_max * sizeof(bool));
	cudaMemset(&dev_current, false, measurable_size_max * sizeof(bool));
	cudaMemset(&dev_target, false, measurable_size_max * sizeof(bool));
	
	cudaMemset(&dev_signal1, false, measurable_size_max * sizeof(bool));
	cudaMemset(&dev_signal2, false, measurable_size_max * sizeof(bool));
	cudaMemset(&dev_measurable, 0, measurable_size_max * sizeof(double));
	cudaMemset(&dev_measurable_old, 0, measurable_size_max * sizeof(double));
}

/*
This function generate other parameter
Input: log type, indicating where this function is invoked
Output: None
*/
void Agent::gen_other_parameters(int P_TYPE){
	Gobserve = new bool[measurable_size_max];
	Gsignal = new bool[measurable_size_max];
	Gload = new bool[measurable_size_max];
	Gmask = new bool[measurable_size_max];
	Gcurrent = new bool[measurable_size_max];
	Gtarget = new bool[measurable_size_max];
	GMeasurable = new double[measurable_size_max];
	GMeasurable_old = new double[measurable_size_max];
	Gprediction = new bool[measurable_size_max];
	Gup = new bool[measurable_size_max];
	Gdown = new bool[measurable_size_max];
	//logging_info->append_log(P_TYPE, "Other Parameter Generated\n");
	//logging::add_CPU_MEM(5 * whole_size * sizeof(bool) + workerSize * sizeof(bool));
	
	cudaMalloc(&dev_observe, measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_signal, measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_load, measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_mask, measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_current, measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_target, measurable_size_max * sizeof(bool));
	
	cudaMalloc(&dev_signal1, measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_signal2, measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_measurable, measurable_size_max * sizeof(double));
	cudaMalloc(&dev_measurable_old, measurable_size_max * sizeof(double));
	//cudaMalloc(&dev_sensor_value, whole_size * sizeof(float));
	//logging_info->append_log(P_TYPE, "GPU Memory Malloced for Other Parameter: " +
		//to_string(5 * whole_size * sizeof(bool) + workerSize * sizeof(bool) + whole_size * sizeof(float)) + "\n");
	//logging::add_GPU_MEM(5 * whole_size * sizeof(bool) + workerSize * sizeof(bool));
}

vector<bool> Agent::getTarget(){
	vector<bool> result;
	for(int i = 0; i < measurable_size; ++i){
		result.push_back(Gtarget[i]);
	}
	return result;
}

void Agent::setTarget(vector<bool> signal){
	for(int i = 0; i < measurable_size; ++i){
		Gtarget[i] = signal[i];
	}
	cudaMemcpy(dev_target, Gtarget, measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
}

vector<vector<double> > Agent::getWeight(){
	cudaMemcpy(Gweights, dev_weights, array_size * sizeof(double), cudaMemcpyDeviceToHost);
	vector<vector<double> > result;
	int n = 0;
	for(int i = 0; i < measurable_size; ++i){
		vector<double> tmp;
		for(int j = 0; j <= i; ++j)
			tmp.push_back(Gweights[n++]);
		result.push_back(tmp);
	}
	return result;
}

vector<double> Agent::getMeasurable(){
	cudaMemcpy(GMeasurable, dev_measurable, measurable_size * sizeof(double), cudaMemcpyDeviceToHost);
	vector<double> result;
	for(int i = 0; i < measurable_size; ++i){
		result.push_back(GMeasurable[i]);
	}
	return result;
}

vector<double> Agent::getMeasurable_old(){
	cudaMemcpy(GMeasurable_old, dev_measurable_old, measurable_size * sizeof(double), cudaMemcpyDeviceToHost);
	vector<double> result;
	for(int i = 0; i < measurable_size; ++i){
		result.push_back(GMeasurable_old[i]);
	}
	return result;
}

vector<vector<bool> > Agent::getMask_amper(){
	vector<vector<bool> > result;
	cudaMemcpy(GMask_amper, dev_mask_amper, mask_amper_size * sizeof(bool), cudaMemcpyDeviceToHost);
	int n = 0;
	for(int i = 0; i < this->sensor_size; ++i){
		vector<bool> tmp;
		for(int j = 0; j <= 2 * i + 1; ++j)
			tmp.push_back(GMask_amper[n++]);
		result.push_back(tmp);
	}
	return result;
}

vector<bool> Agent::getUp(){
	vector<bool> result;
	for(int i = 0; i < measurable_size; ++i){
		result.push_back(Gup[i]);
	}
	return result;
}

vector<bool> Agent::getDown(){
	vector<bool> result;
	for(int i = 0; i < measurable_size; ++i){
		result.push_back(Gdown[i]);
	}
	return result;
}

void Agent::amperand(vector<vector<double> > &weights, vector<double> &measurable, vector<double> &measurable_old, int idx1, int idx2, double total, bool merge, int LOG_TYPE){
	vector<double> result1, result2;

	this->logging_info->append_log(logging::AMPERAND, "[AMPERAND]:\n");
	this->logging_info->add_indent();
	
	this->logging_info->append_log(logging::AMPERAND, "Amper Index: " + to_string(idx1) + ", " + to_string(idx2) + "\n");

	for(int i = 0; i < measurable.size(); ++i){
		if(i == idx1 || i == idx2){
			result1.push_back(weights[idx2][idx1]);
			result1.push_back(0);
		}
		else{
			result1.push_back(weights[idx2][idx1] * measurable[i] / total);
			result1.push_back(weights[idx2][idx1] * (1 - measurable[i] / total));
		}
	}
	result1.push_back(result1[0] + result1[1]);
	
	//first row
	for(int i = 0; i < measurable.size(); ++i){
		if(i == idx1){
			result2.push_back(weights[compi(idx2)][idx1]);
			result2.push_back(weights[idx2][compi(idx1)] + weights[compi(idx2)][compi(idx1)]);
		}
		else if(i == idx2){
			result2.push_back(weights[idx2][compi(idx1)]);
			result2.push_back(weights[compi(idx2)][idx1] + weights[compi(idx2)][compi(idx1)]);
		}
		else{
			result2.push_back((1 - weights[idx2][idx1]) * measurable[i] / total);
			result2.push_back((1 - weights[idx2][idx1]) * (1 - measurable[i] / total));
		}
	}
	result2.push_back(0);
	result2.push_back(result2[0] + result2[1]);

	if(merge){//if it is the new row that need to append without pop
		weights.push_back(result1);
		weights.push_back(result2);
		measurable.push_back(result1.back());
		measurable.push_back(result2.back());
		measurable_old.push_back(result1.back());
		measurable_old.push_back(result2.back());

		this->logging_info->append_log(logging::AMPERAND, "New Sensor Merged in Temp Variable, Measurable Size: " + to_string(measurable.size()) + ")\n");
		//just put in two new vector and two cc value
	}
	else{//else need to pop the old value and push in the new one, need t get rid of corresponding value
		int old_idx1 = measurable.size() - 2, old_idx2 = measurable.size() - 1;
		result1.erase(result1.begin() + old_idx1);
		result1.erase(result1.begin() + old_idx2);
		result2.erase(result2.begin() + old_idx1);
		result2.erase(result2.begin() + old_idx2);
		weights.pop_back();weights.pop_back();
		weights.push_back(result1);
		weights.push_back(result2);
		measurable.pop_back();measurable.pop_back();
		measurable_old.pop_back();measurable_old.pop_back();
		measurable.push_back(result1.back());
		measurable.push_back(result2.back());
		measurable_old.push_back(result1.back());
		measurable_old.push_back(result2.back());

		this->logging_info->append_log(logging::AMPERAND, "New Sensor Replace in Temp Variable, Measurable Size: " + to_string(measurable.size()) + ")\n");
	}
	//second row

	this->logging_info->reduce_indent();
	this->logging_info->append_process(logging::AMPERAND, LOG_TYPE);
}

void Agent::generate_delayed_weights(vector<vector<double> > &weights, vector<double> &measurable, vector<double> &measurable_old, double last_total, int measurable_id, bool merge, int LOG_TYPE){
	vector<double> result1, result2;

	this->logging_info->append_log(logging::GENERATE_DELAY, "[GENERATE DELAY SENSOR]:\n");
	this->logging_info->add_indent();

	int delay_sensor_id1 = measurable_id, delay_sensor_id2 = compi(measurable_id);

	this->logging_info->append_log(logging::GENERATE_DELAY, "Delay Index: " + to_string(delay_sensor_id1) + ", " + to_string(delay_sensor_id2) + "\n");

	for(int i = 0; i < measurable.size() - 2 + 2 * merge; ++i){
		result1.push_back(measurable_old[delay_sensor_id1] * measurable[i] / last_total);
	}
	result1.push_back(result1[0] + result1[1]);
	//row 1
	for(int i = 0; i < measurable.size() - 2 + 2 * merge; ++i){
		result2.push_back(measurable_old[delay_sensor_id2] * measurable[i] / last_total);
	}
	result2.push_back(0);
	result2.push_back(result2[0] + result2[1]);
	//row 2

	if(!merge){
		weights.pop_back();weights.pop_back();
	}
	weights.push_back(result1);
	weights.push_back(result2);

	if(!merge){
		measurable.pop_back();measurable.pop_back();
		measurable_old.pop_back();measurable_old.pop_back();
	}
	measurable.push_back(result1.back());
	measurable.push_back(result2.back());
	measurable_old.push_back(result1.back());
	measurable_old.push_back(result2.back());

	this->logging_info->reduce_indent();
	this->logging_info->append_process(logging::GENERATE_DELAY, LOG_TYPE);
}

double Agent::get_delta_weight_sum(vector<bool> signal){
	float result = 0;
	float *dev_result;
	cudaMalloc(&dev_result, sizeof(float));
	cudaMemset(dev_result, 0, sizeof(float));
	
	for(int i = 0; i < signal.size(); ++i) Gsignal[i] = signal[i];

	delta_weight_sum_kernel<<<(sensor_size + 255) / 256, 256>>>(dev_measurable, Gsignal, dev_result, sensor_size);

	cudaMemcpy(&result, dev_result, sizeof(double), cudaMemcpyDeviceToHost);

	return result;
}