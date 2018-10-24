#include "AgentDM.h"

AgentDM::AgentDM(Agent *agent){
	this->agent=agent;
}

AgentDM::~AgentDM(){}

bool AgentDM::writeData(string filename){
	agent->logging_info->append_log(logging::SAVING, "[Data Saving]:\n");
	agent->logging_info->add_indent();
	agent->logging_info->append_log(logging::SAVING, "Saving Data to File: " + filename+"\n");

	ofstream OutFile;
	int measurable_size=agent->measurable_size;
	//agent->copy_Data_To_CPU();
	OutFile.open(filename+"_"+agent->name, ios::out | ios::binary);
	string *sensors_names = new string[agent->sensors_names.size()];
	for(int i = 0; i<agent->sensors_names.size(); ++i) sensors_names[i] = agent->sensors_names[i];

	OutFile.write((char*)agent->Gdir, measurable_size*measurable_size*sizeof(bool));
	agent->logging_info->append_log(logging::SAVING, "Saving Direction Data, Size of Data: "
		+ to_string(measurable_size * measurable_size * sizeof(bool)) + " Bytes(" + to_string
		(measurable_size * measurable_size * sizeof(bool) / (1024.0 * 1024)) + "M)\n");

	OutFile.write((char*)agent->Gweights, measurable_size*measurable_size*sizeof(double));
	agent->logging_info->append_log(logging::SAVING, "Saving Weights Data, Size of Data: "
		+ to_string(measurable_size * measurable_size * sizeof(double)) + " Bytes(" + to_string
		(measurable_size * measurable_size * sizeof(double) / (1024.0 * 1024)) + "M)\n");

	OutFile.write((char*)agent->Gthresholds, measurable_size*measurable_size*sizeof(double));
	agent->logging_info->append_log(logging::SAVING, "Saving Thresholds Data, Size of Data: "
		+ to_string(measurable_size * measurable_size * sizeof(double)) + " Bytes(" + to_string
		(measurable_size * measurable_size * sizeof(double) / (1024.0 * 1024)) + "M)\n");

	OutFile.write((char*)sensors_names, agent->sensors_names.size()*sizeof(string));

	delete[] sensors_names;
	OutFile.close();
	agent->logging_info->append_log(logging::SAVING, "Data Saved\n");
	agent->logging_info->append_process(logging::SAVING, logging::PROCESS);
	agent->logging_info->reduce_indent();

	return true;
}

bool AgentDM::readData(string filename){
	int measurable_size=agent->measurable_size;
	//string *sensors_names = new string[agent->sensors_names.size()];
	fstream InFile(filename+"_"+agent->name, ios::binary | ios::in);
	agent->logging_info->append_log(logging::LOADING, "[Data Loading]:\n");
	agent->logging_info->add_indent();
	agent->logging_info->append_log(logging::LOADING, "File Read for Input, File Name: "+filename+"\n");

	InFile.read((char *)agent->Gdir, measurable_size * measurable_size*sizeof(bool));
	agent->logging_info->append_log(logging::LOADING, "Read Direction Data, Size of Data: "
		+ to_string(measurable_size * measurable_size * sizeof(bool)) + " Bytes(" + to_string
		(measurable_size * measurable_size * sizeof(bool) / (1024.0 * 1024)) + "M)\n");
	InFile.read((char *)agent->Gweights, measurable_size * measurable_size*sizeof(double));
	agent->logging_info->append_log(logging::LOADING, "Read Weights Data, Size of Data: "
		+ to_string(measurable_size * measurable_size * sizeof(double)) + " Bytes(" + to_string
		(measurable_size * measurable_size * sizeof(double) / (1024.0 * 1024)) + "M)\n");
	InFile.read((char *)agent->Gthresholds, measurable_size * measurable_size*sizeof(double));
	agent->logging_info->append_log(logging::LOADING, "Read Thresholds Data, Size of Data: "
		+ to_string(measurable_size * measurable_size * sizeof(double)) + " Bytes(" + to_string
		(measurable_size * measurable_size * sizeof(double) / (1024.0 * 1024)) + "M)\n");

	InFile.close();
	agent->logging_info->append_log(logging::LOADING, "File Read Complete\n");
	agent->logging_info->append_process(logging::LOADING, logging::INIT);
	agent->logging_info->reduce_indent();

	agent->logging_info->append_log(logging::COPY, "[Matrix Copy]:\n");
	agent->logging_info->add_indent();

	cudaMemcpy(agent->dev_weights, agent->Gweights, measurable_size * measurable_size * sizeof(double), cudaMemcpyHostToDevice);
	agent->logging_info->append_log(logging::COPY, "Weight Matrix Copied to GPU\n");
	cudaMemcpy(agent->dev_dir, agent->Gdir, measurable_size * measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
	agent->logging_info->append_log(logging::COPY, "Direction Matrix Copied to GPU\n");
	cudaMemcpy(agent->dev_thresholds, agent->Gthresholds, measurable_size * measurable_size * sizeof(double), cudaMemcpyHostToDevice);
	agent->logging_info->append_log(logging::COPY, "Thresholds Matrix Copied to GPU\n");

	//in.read((char *)sensors_names, measurable_size*sizeof(string));
	//for now do not read sensor names, do intersection in the future
	agent->logging_info->append_process(logging::COPY, logging::INIT);
	agent->logging_info->reduce_indent();
	return true;
}