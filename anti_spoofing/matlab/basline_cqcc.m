%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ASVspoof 2017 CHALLENGE:
% Audio replay detection challenge for automatic speaker verification anti-spoofing
%
% http://www.spoofingchallenge.org/
%
% ====================================================================================
% Matlab implementation of the baseline system for replay detection based
% on constant Q cepstral coefficients (CQCC) features + Gaussian Mixture Models (GMMs)
% ====================================================================================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;

% set paths to the wave files and protocols
%pathToDatabase = fullfile('..','ASVspoof2017_train_dev','flac');
pathToDatabase = fullfile('..','data','LA');

data_type = "dev";

switch data_type
    case {"train"}
        protocolFile = fullfile(pathToDatabase, 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.train.trn.txt');
    case {"eval"}
        protocolFile = fullfile(pathToDatabase, 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.eval.trl.txt');
    case {"dev"}
        protocolFile = fullfile(pathToDatabase, 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.dev.trl.txt');
end

disp(protocolFile);
% read train protocol
fileID = fopen(protocolFile);
protocol = textscan(fileID, '%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{2};
labels = protocol{5};

B = 96;
fmax = 16000/2;
fmin = fmax/2^9;
d = 16;
cf = 19;
ZsdD = "Zs";


% get indices of genuine and spoof files
genuineIdx = find(strcmp(labels, 'bonafide'));
spoofIdx = find(strcmp(labels, 'spoof'));

%% Feature extraction for training data

% extract features for BONAFIDE training data and store in cell array
disp(strcat('Extracting features for BONAFIDE ', data_type, ' data...'));
genuineFeatureCell = cell(size(genuineIdx));

params_choices = strcat(data_type, "_", int2str(B), "_", int2str(d), "_", int2str(cf), "_", ZsdD);

if ~exist(fullfile('../data/features/', params_choices), 'dir')
   mkdir(fullfile('../data/features/', params_choices));
end

f = waitbar(0., strcat('Extracting features for BONAFIDE  ', data_type, ' data...'));
len = length(genuineIdx);
parfor i=1:len
    
    save_name = strrep(filelist{genuineIdx(i)}, '.flac', '_cqcc.mat');
    save_path = fullfile('../data/features/', params_choices, save_name);

    filePath = fullfile(pathToDatabase, strcat('ASVspoof2019_LA_', data_type), 'flac', strcat(filelist{genuineIdx(i)}, ".flac"));
    [x, fs] = audioread(filePath);
 
    waitbar(1.-i/len, f);

    tmp_fea = cqcc(x, fs, B, fmax, fmin, d, cf, ZsdD);

    genuineFeatureCell{i} = tmp_fea
    parsave(save_path, tmp_fea)
end
close(f);
disp('Done!');

% extract features for SPOOF training data and store in cell array
disp(strcat('Extracting features for SPOOF ', data_type, ' data...'));
spoofFeatureCell = cell(size(spoofIdx));

f = waitbar(0., strcat('Extracting features for SPOOF ', data_type, ' data...'));
len = length(spoofIdx);
parfor i=1:len
    save_name = strrep(filelist{spoofIdx(i)}, '.flac', '_cqcc.mat');
    save_path = fullfile('../data/features/', params_choices, save_name);

    filePath = fullfile(pathToDatabase, strcat('ASVspoof2019_LA_', data_type), 'flac', strcat(filelist{spoofIdx(i)}, ".flac"));
    [x, fs] = audioread(filePath);
    tmp_fea = cqcc(x, fs, B, fmax, fmin, d, cf, ZsdD);
    
    waitbar(1.-i/len, f);

    spoofFeatureCell{i} = tmp_fea
    parsave(save_path, tmp_fea)
end
close(f);
disp('Done!');

function parsave(fname, x)
    save(fname, 'x', '-v6')
end