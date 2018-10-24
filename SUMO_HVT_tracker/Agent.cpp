#include "Agent.h"
/*
----------------Agent Base Class-------------------
*/
Agent::Agent(int type, bool using_agent){
	Gdir = NULL;
	Gweights = NULL;
	Gthresholds = NULL;
	GMask_amper = NULL;

	dev_dir = NULL;
	dev_weights = NULL;
	dev_thresholds = NULL;
	dev_mask_amper = NULL;

	this->type = type;
	this->threshold = 0.25;
	
	this->using_agent = using_agent;
	this->last_total = 0;
	this->total = 0;
	this->memory_expansion = 0.5;

	//agentDM=new AgentDM(this);
}

Agent::~Agent(){}

static int randInt(int n){
	return rand()%n;
}

//float Agent::decide(vector<bool> signal, double phi, bool active, double q){//the decide function //SIQI: I've remove the "double q" input.
float Agent::decide(vector<bool> signal, double phi, bool active){//the decide function
	this->phi = phi;
	//SIQI:this->q = q;
	setSignal(signal);
	update_state_GPU(active);
	//logging::num_sim++;

	halucinate_GPU();
	return distance_big(dev_load, dev_target);
	//calculate distance

	//TBD
}

vector<string> Agent::getDecision(){
	return decision;
}

void Agent::init_sensors_name(vector<string> &sensors_names, int P_TYPE){
	this->sensors_names = sensors_names;
	//logging_info->append_log(P_TYPE, "Agent Sensors Name From Python, Size: " + to_string(sensors_names.size()) + "\n");
}

void Agent::init_size(int sensor_size, int LOG_TYPE){
	this->sensor_size = sensor_size;
	this->measurable_size = 2 * sensor_size;
	this->array_size = measurable_size * (measurable_size + 1) / 2;
	this->mask_amper_size = sensor_size * (sensor_size + 1);
	
	this->sensor_size_max = (int)(this->sensor_size * (1 + this->memory_expansion));
	this->measurable_size_max = 2 * sensor_size_max;
	this->array_size_max = measurable_size_max * (measurable_size_max + 1) / 2;
	this->mask_amper_size_max = this->sensor_size_max * (this->sensor_size_max + 1);
	
	logging_info->append_log(logging::SIZE, "[SIZE]:\n");
	logging_info->add_indent();

	logging_info->append_log(logging::SIZE, "Sensor Size: " + to_string(this->sensor_size) +
		"(Max Size:" + to_string(this->sensor_size_max) + ")\n");
	logging_info->append_log(logging::SIZE, "Measurable Size: " + to_string(this->measurable_size) +
		"(Max Size:" + to_string(this->measurable_size_max) + ")\n");
	logging_info->append_log(logging::SIZE, "Array Size: " + to_string(this->array_size) +
		"(Max Size:" + to_string(this->array_size_max) + ")\n");
	logging_info->append_log(logging::SIZE, "Mask Size: " + to_string(this->mask_amper_size) +
		"(Max Size:" + to_string(this->mask_amper_size_max) + ")\n");

	logging_info->append_process(logging::SIZE, LOG_TYPE);
	logging_info->reduce_indent();
}

void Agent::init_name(string name, int LOG_TYPE){
	this->name = name;
	logging_info->append_log(LOG_TYPE, "Agent Name From Python: " + name+"\n");
}

void Agent::reset_time(int n, int LOG_TYPE){
	this->t = n;
	logging_info->append_log(LOG_TYPE, "Time Reset to: " + to_string(n) + "\n");
	srand(time(NULL));
	logging_info->append_log(LOG_TYPE, "Time Seed Set \n");
}

void Agent::init_data(string name, int sensor_size, vector<string> sensors_names, string filename){
	//data init
	this->logging_info = new logging(name);
	//will put this into python as mix object in the end, should be an object across border
	logging_info->append_log(logging::INIT, "[Data Initialization]:\n");
	logging_info->add_indent();
	
	reset_time(0, logging::INIT);

	logging_info->append_log(logging::INIT, "Agent Type : DISCOUNTED\n");
	init_name(name, logging::INIT);
	this->base_sensor_size = sensor_size;
	init_size(sensor_size, logging::INIT);
	//copy_sensors_name(sensors_names, logging::INIT);
	
	//logging_info->append_log(logging::MALLOC,"[Matrix Allocation:]\n");
	//logging_info->add_indent();
	gen_weight(logging::MALLOC);
	gen_direction(logging::MALLOC);
	gen_thresholds(logging::MALLOC);
	gen_mask_amper(logging::MALLOC);
	gen_other_parameters(logging::INIT);

	//logging_info->append_log(logging::COPY, "[Matrix Copy]:\n");
	//logging_info->add_indent();
	init_weight(logging::COPY);
	init_direction(logging::COPY);
	init_thresholds(logging::COPY);
	init_mask_amper(logging::COPY);
	init_other_parameter(logging::MALLOC);
	//logging_info->append_process(logging::COPY, logging::INIT);
	//logging_info->reduce_indent();
	//for now do not save worker infomation

	logging_info->append_process(logging::INIT, logging::PROCESS);
	logging_info->reduce_indent();
}

//those three functions down there are get functions for the variable in C++
vector<bool> Agent::getCurrent(){
	vector<bool> result;
	for(int i = 0; i < measurable_size; ++i){
		result.push_back(Gcurrent[i]);
	}
	return result;
}

vector<bool> Agent::getPrediction(){
	vector<bool> result;
	for(int i = 0; i < measurable_size; ++i){
		result.push_back(Gprediction[i]);
	}
	return result;
}

vector<bool> Agent::getSignal(){
	vector<bool> result;
	for(int i = 0; i < measurable_size; ++i){
		result.push_back(Gsignal[i]);
	}
	return result;
}

vector<bool> Agent::getLoad(){
	vector<bool> result;
	for(int i = 0; i < measurable_size; ++i){
		result.push_back(Gload[i]);
	}
	return result;
}

logging Agent::get_log(){
	logging_info->finalize_log();
	return *logging_info;
}

void Agent::savingData(string filename){
	agentDM->writeData(filename);
}

void Agent::reallocate_memory(int target_sensor_size, int LOG_TYPE){
	this->logging_info->append_log(logging::REMALLOC, "[REMALLOC:]\n");
	this->logging_info->add_indent();

	free_other_parameters();
	init_size(target_sensor_size, logging::REMALLOC);

	gen_weight(logging::REMALLOC);
	gen_direction(logging::REMALLOC);
	gen_thresholds(logging::REMALLOC);
	gen_mask_amper(logging::REMALLOC);
	gen_other_parameters(logging::REMALLOC);

	init_weight(logging::REMALLOC);
	init_direction(logging::REMALLOC);
	init_thresholds(logging::REMALLOC);
	init_mask_amper(logging::REMALLOC);
	init_other_parameter(logging::REMALLOC);

	this->logging_info->reduce_indent();
	this->logging_info->append_process(logging::REMALLOC, LOG_TYPE);
}

void Agent::copy_weight(int P_TYPE, int start_idx, int end_idx, vector<double> &measurable, vector<double> &measurable_old, vector<vector<double> > &weights, vector<vector<bool> > &mask_amper){
	int measurable_start = start_idx, measurable_end = end_idx;
	int weight_start = measurable_start * (measurable_start + 1) / 2;
	int weight_end = measurable_end * (measurable_end + 1) / 2;
	int amper_start = (start_idx / 2) * (start_idx / 2 + 1);
	int amper_end = (end_idx / 2) * (end_idx / 2 + 1);
	if(measurable_end > measurable_size_max){
		//record error message in log
		return;
	}
	for(int i = measurable_start; i < measurable_end; ++i){
		GMeasurable[i] = measurable[i];
		GMeasurable_old[i] = measurable_old[i];
	}
	cudaMemcpy(dev_measurable + measurable_start, GMeasurable + measurable_start, (measurable_end - measurable_start) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_measurable_old + measurable_start, GMeasurable_old + measurable_start, (measurable_end - measurable_start) * sizeof(double), cudaMemcpyHostToDevice);

	int x = 0, y = measurable_start;
	for(int i = weight_start; i < weight_end; ++i){
		Gweights[i] = weights[y][x++];
		if(x > y){
			x = 0;
			y++;
		}
	}
	cudaMemcpy(dev_weights + weight_start, Gweights + weight_start, (weight_end - weight_start) * sizeof(double), cudaMemcpyHostToDevice);
	
	x = 0; y = start_idx / 2;
	for(int i = amper_start; i < amper_end; ++i){
		GMask_amper[i] = mask_amper[y][x++];
		if(x > 2 * y + 1){
			x = 0;
			y++;
		}
	}
	cudaMemcpy(dev_mask_amper + amper_start, GMask_amper + amper_start, (amper_end - amper_start) * sizeof(bool), cudaMemcpyHostToDevice);
}

void Agent::pruning(vector<bool> signal){
	vector<vector<double> > weights = getWeight();
	vector<double> measurable = getMeasurable();
	vector<double> measurable_old = getMeasurable_old();
	vector<vector<bool> > mask_amper = getMask_amper();

	//logging
	//pruning for meaurable, measurable_old and weights
	int col_escape = 0;
	for(int i = 0; i < weights.size(); ++i){
		if(signal[i]) col_escape++;//need to escape row
		else{
			int row_escape = 0;
			for(int j = 0; j < weights[i].size(); ++j){
				if(signal[j]) row_escape++;//itself need to be removed
				else weights[i - col_escape][j - row_escape] = weights[i][j];
			}
			measurable[i - col_escape] = measurable[i];
			measurable_old[i - col_escape] = measurable_old[i];

			for(int j = 0; j < row_escape; ++j) weights[i].pop_back();
		}
	}
	for(int i = 0; i < col_escape; ++i){
		weights.pop_back();
		measurable.pop_back();
		measurable_old.pop_back();
	}
	//pruning for mask_amper
	col_escape = 0;
	for(int i = 0; i < mask_amper.size(); ++i){
		if(signal[2 * i]) col_escape++;
		else{
			int row_escape = 0;
			for(int j = 0; j < mask_amper[i].size(); ++j){
				if(signal[j]) row_escape++;//itself need to be removed
				else weights[i - col_escape][j - row_escape] = weights[i][j];
			}

			for(int j = 0; j < row_escape; ++j) weights[i].pop_back();
		}
	}
	for(int i = 0; i < col_escape; ++i){
		mask_amper.pop_back();
	}
	//write back to GPU and change matrix size
	copy_weight(logging::INIT, measurable_size, measurable.size(), measurable, measurable_old, weights, mask_amper);
	this->sensor_size = measurable.size() / 2;
	this->measurable_size = measurable.size();
	this->array_size = this->measurable_size * (this->measurable_size + 1) / 2;
	this->mask_amper_size = this->sensor_size * (this->sensor_size + 1);
}

void Agent::amper(vector<vector<bool> > lists, int LOG_TYPE){
	vector<vector<double> > weights = getWeight();
	vector<double> measurable = getMeasurable();
	vector<double> measurable_old = getMeasurable_old();
	vector<vector<bool> > mask_amper = getMask_amper();

	this->logging_info->append_log(logging::AMPER, "[AMPER]:\n");

	for(int i = 0; i < lists.size(); ++i){
		vector<int> list;
		for(int j = 0; j < lists[i].size(); ++j){
			if(lists[i][j]) list.push_back(j);
		}
		if(list.size() < 2) continue;//probably need sth to append in error.log
		amperand(weights, measurable, measurable_old, list[0], list[1], total, true, LOG_TYPE);
		for(int j = 2; j < list.size(); ++j){
			amperand(weights, measurable, measurable_old, list[j], measurable.size() - 2, total, false, LOG_TYPE);
		}

		int s = mask_amper.back().size() + 2;
		vector<bool> tmp;
		for(int j = 0; j < s; ++j){
			tmp.push_back(lists[i][j]);
		}
		mask_amper.push_back(tmp);
	}
	
	if(measurable.size() > measurable_size_max){//need to reallocate
		reallocate_memory(measurable.size() / 2, logging::AMPER);
		copy_weight(logging::INIT, 0, measurable.size(), measurable, measurable_old, weights, mask_amper);
	}
	else{//just copy to the back
		copy_weight(logging::INIT, measurable_size, measurable.size(), measurable, measurable_old, weights, mask_amper);
		this->sensor_size = measurable.size() / 2;
		this->measurable_size = measurable.size();
		this->array_size = this->measurable_size * (this->measurable_size + 1) / 2;
		this->mask_amper_size = this->sensor_size * (this->sensor_size + 1);
	}
}

void Agent::delay(vector<vector<bool> > lists){
	vector<vector<double> > weights = getWeight();
	vector<double> measurable = getMeasurable();
	vector<double> measurable_old = getMeasurable_old();
	vector<vector<bool> > mask_amper = getMask_amper();
	this->logging_info->append_log(logging::DELAY, "[DELAY]:(t=" + to_string(this->t) + ")\n");
	this->logging_info->add_indent();
	int success_amper = 0;

	this->logging_info->append_log(logging::DELAY, "Init Measurable Size: " + to_string(measurable.size()) + "\n");
	
	for(int i = 0; i < lists.size(); ++i){
		vector<int> list;
		for(int j = 0; j < lists[i].size(); ++j){
			if(lists[i][j]) list.push_back(j);
		}
		if(list.size() == 0) continue;//probably need sth to append in error.log
		success_amper++;
		
		if(list.size() != 1){
			amperand(weights, measurable, measurable_old, list[0], list[1], total, true, logging::DELAY);
			for(int j = 2; j < list.size(); ++j){
				amperand(weights, measurable, measurable_old, list[j], measurable.size() - 2, total, false, logging::DELAY);
			}
			generate_delayed_weights(weights, measurable, measurable_old, last_total, weights.size() - 2, false, logging::DELAY);
		}
		else{
			generate_delayed_weights(weights, measurable, measurable_old, last_total, list[0], true, logging::DELAY);
		}
		
		int s = mask_amper.back().size() + 2;
		vector<bool> tmp;
		for(int j = 0; j < s; ++j){
			tmp.push_back(lists[i][j]);
		}
		mask_amper.push_back(tmp);
	}

	this->logging_info->append_log(logging::DELAY, "Amper Done: " + to_string(success_amper) + "("+ to_string(lists.size()) + ")\n");

	if(measurable.size() > measurable_size_max){//need to reallocate
		reallocate_memory(measurable.size() / 2, logging::DELAY);
		copy_weight(logging::INIT, 0, measurable.size(), measurable, measurable_old, weights, mask_amper);
	}
	else{//just copy to the back
		copy_weight(logging::INIT, measurable_size, measurable.size(), measurable, measurable_old, weights, mask_amper);
		this->sensor_size = measurable.size() / 2;
		this->measurable_size = measurable.size();
		this->array_size = this->measurable_size * (this->measurable_size + 1) / 2;
		this->mask_amper_size = this->sensor_size * (this->sensor_size + 1);
	}

	this->logging_info->reduce_indent();
	this->logging_info->append_process(logging::DELAY, logging::PROCESS);
}

/*
----------------Agent Base Class-------------------
*/

/*
----------------Agent_Discounted Class-------------------
*/

Agent_Discounted::Agent_Discounted(double q,  bool using_agent)
	:Agent(DISCOUNTED, using_agent){
	this->q = q;
}

Agent_Discounted::~Agent_Discounted(){}

/*
----------------Agent_Discounted Class-------------------
*/
