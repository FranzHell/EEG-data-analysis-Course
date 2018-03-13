%% ERP Analysis Script for EEG Course 2015 - RESAMPLING SCRIPT
% Franz Hell - December, 2015


clear all;  %% clear current workspace
close all; %% close all windows


%% 1. Load data into Matlab
% specify specific file location
% load raw data - structure
% contains some additional information

% for most data format either matlab toolboxes or scripts are available to
% transfer them from their native data format to a matlab structure;

  load('E:\course2016EEG\dataset1.mat')
  load('EEGlabstructure.mat')
  lala =struct2cell(EEG.chanlocs)
Channelnames = squeeze(lala(1,:,:));
Channelnames(66) = {'STNL'};
Channelnames(67) = {'STNR'};
  Fs = 422; % this is the sampling frequency for this dataset
 data_EEG_sync = data_EEG_sync*1000; %% scale the data to match the LFP dataset


%% 2. Explore the data
%plot raw data from channel Oz and EOG Channel, use xlabel(str), ylabel(str) and
%title(str) to label graph and axis

% create time vector
Time =1/Fs:1/Fs:length(data_EEG_sync)/Fs;   

figure(1)

subplot(2,1,1)
plot(Time,data_EEG_sync(67,:));
title('raw EEG on Channel STN left ' )
xlabel('time')
ylabel('Amplitude[V]')

subplot(2,1,2)
plot(Time,data_EEG_sync(1,:));
title('raw EEG on Channel Fp1' )
xlabel('time')
ylabel('Amplitude[V]')


% explore some channels, plot only segments to zoom in

figure
for i = 1:16
  
    subplot(8,2,i)
    plot(Time(10000:30000),data_EEG_sync(i,10000:30000))
%     hold on
%    title(strcat('raw EEG on Channel ', Channel(1, i).Name))
   axis tight 
end
%% 2.1 Exercise: creating a simple "butterfly plot"
% create a plot in which every trace is shown in the same graph,

figure
plot(Time(10000:30000),data_EEG_sync(:,10000:30000))

%% 2.2 Exercise: creating a simple "butterfly plot" which lets you scroll through time
% use a while loop that goes from the beginning to the end of the data, and
% update the timeframe you are plotting



%% 3. Filtering
% do some filtering to remove slow drifts (high pass), electrical noise (notch filter at 50Hz) 
%and frequencies over 100Hz (including electrical noise harmonics) construct filters first

%if you are unsure which parameters and filters to use, type "fdatool". if
%you have an up-to-date matlab you can design your own filter in a gui and
%export the filter via file . generate matlab code

%take a look at this video: 
% http://de.mathworks.com/videos/digital-filtering-97062.html on the
%right side of this webpage you will find more videos that introduce you
%to signal analysis with Matlab

%highpassfilter with cutoff 1Hz and order 2
cutoffhigh = 1;
[b,a] = butter(2, cutoffhigh/(Fs/2), 'high');
figure
freqz(b,a)
%lowpassfilter with cutoff at 100Hz and order 2
cutofflow = 20;
[c,d] = butter(2, cutofflow/(Fs/2));
figure
freqz(d,c)
%notchfilter at 50Hz and order 2
notchborderlow = 45;
notchborderhigh = 55;
[e,f] = butter(2,[notchborderlow*2/Fs,notchborderhigh*2/Fs],'stop'); 
figure
freqz(f,e)

%bandpassfilterbetween 45 and 55Hz and order 2
[g,h] = butter(2,[45*2/Fs,55*2/Fs],'bandpass'); 


%% 3.1 Apply a filter
% apply a filter to a channel of your choice, look at it in the same plot
%before and after filtering; 
%there are many filter functions in matlab,
%filtfilt (see doc filtfilt) e.g. does avoid phase disturbations by processing the input data
%in both the forward and reversedirections.
%this might be useful when you want to look at phase relations like in phase-phase or phase-amplitude coupling analysis

dataOz= data_EEG_sync(30,:);
dataOz_highpassfiltered = filtfilt(b,a,dataOz);
figure
plot(dataOz(8000:11000),'b')
hold on
plot(dataOz_highpassfiltered(8000:11000),'r')

%% 3.2 Exercise 2: Try out some parameters
% try out the other filter and play around with some parameters.
% look at the data before and after filtering to see if you have achieved
% what you wanted and if the parameters are chosen properly


%% 3.3 Exercise
%there is another filter function called filter, check out the documentation (doc) both of them and
%see how this affects your end results


%% 3.4 Exercise 3: Filter all EEGdata with all filters
%hit all channels with all of the  before constructed filters.
%this should remove drifts (highpassfilter) from e.g. sweating, line noise
%(at 50Hz) and high frequencies.
%
%initialize new empty matrix
electrodescount = 65;
data_filtered = zeros(electrodescount,length(data_EEG_sync));

% filter raw data from all channels with all the before constructed filters

for i = 1:electrodescount
data_filtered2(i,:) = filtfilt(b,a,data_EEG_sync(i,:));
data_filtered2(i,:) = filtfilt(c,d,data_filtered2(i,:));
data_filtered2(i,:) = filtfilt(e,f,data_filtered2(i,:));
end  

data_EEG_sync(1:65,:) = data_filtered2;



%% 3.5 Scroll through data
% Scroll through data:
% take another look at the data before and after filtering
% Contruct a loop that lets you scroll through the data

an = 1; 
len_data= length(data_EEG_sync(2,:));
figure;
                en = an - 1 + 2500;  
                while en < len_data
                                                                           % plot data
                    plot(Time(1,an:en),data_EEG_sync(2,an:en))
                    hold on
                    plot(Time(1,an:en),data_EEG_sync(2,an:en),'g');
                    xlabel('Time in Seconds')
                    hold off
 [~,~] = ginput(1);
                    an = an + 2500;
                    en = an - 1 + 2500;
                end

% for those who have time: contruct a loop that lets you scroll through more than one channel
% by e.g. using a for loop over channels and subplots for the different
% channels


% update trialinfo and matrxes                     
%% 4. Epoching

% cut out the epochs in the data, do baseline correction for all threee trialtypes - done sequentially in a script 
% -  writing a function that handles different types of input will shorten
% the amount of code 



epochdur = 422;
TimeX = -epochdur/Fs:1/Fs:epochdur/Fs;

ll=1
for i=1:length(triggers)
store_StimlockedEEG(ll,:,:) = data_EEG_sync(:,round(triggers(i))-epochdur:round(triggers(i))+epochdur);
ll=ll+1;

end


% exercise create a response locked trial dataset
clearvars data_filtered data_filtered2; %% keep your workspace clean, otherwise your RAM will overflow and everything will be painful slow.



%% 5. Averageing
% calculate and plot the average
meanFP1 = squeeze(sum(store_StimlockedEEG(:,1,:))/size(store_StimlockedEEG(:,1,:),1));


figure;
subplot(2,1,1)
plot(TimeX, meanFP1)
subplot(2,1,2)
plot(TimeX,squeeze(mean(store_StimlockedEEG(:,1,:))))             

% look at an ERP at different sites

figure;
df=1;
n=4;
for i = [1,30,66,67]

subplot(n,1,df)
df=df+1;
plot(TimeX,squeeze(mean(store_StimlockedEEG(:,i,:))))
xlabel('Time [s]')
title(strcat('Channel - ', Channelnames(i)))
hold on
line([0 0],[min(squeeze(mean(store_StimlockedEEG(:,1,:)))) max(squeeze(mean(store_StimlockedEEG(:,1,:))))],'Color','g') %onset time target
axis tight

end


%% 5.1 Exercise: Compare ERPs from different channels
%draw ERPs from different channels into the same plot

idxSC = find(strcmp(info2(:,1)', 'SC'))
idxSI = find(strcmp(info2(:,1)', 'SI'))
idxRI = find(strcmp(info2(:,1)', 'RI'))
RTsx = cell2mat(info2(:,4));
[idxquick,c] = find(RTsx < mean(RTsx) & mean(RTsx) > .3 )
[idxslow,h] = find(RTsx > mean(RTsx) & mean(RTsx) < 2 )

%% 5.2 Exercise: Compare ERPs from different conditions
% look at an ERP in different conditions at the same channel
TimeX = -epochdur/Fs:1/Fs:epochdur/Fs
figure;
subplot(2,1,1)
plot(TimeX,squeeze(mean(store_StimlockedEEG(idxquick,1,:))))
hold on
plot(TimeX,squeeze(mean(store_StimlockedEEG(idxslow,1,:))))
xlabel('Time [s]')
hold on
axis tight
% draw ERPs from different conditions into the same plot, and compare them
% calculate the difference between two mean timeseries from different
% conditions, use subplots to put ERPs and differences on the same figure
%% 5.3 Topographical representation of the ERP
windowk = 10;
ddd = ginput(1)
ggg = ginput(1)
%%%
% look at the topographical representation, the topplot
subplot(2,2,3)
topoplotIndie(mean(squeeze(mean(store_StimlockedEEG(idxslow,1:65,ceil(ddd(1)*Fs)-windowk:ceil(ddd(1)*Fs)+windowk))),2)', EEG.chanlocs)
subplot(2,2,4)

topoplotIndie(mean(squeeze(mean(store_StimlockedEEG(idxslow,1:65,ceil(ggg(1)*Fs)-windowk:ceil(ggg(1)*Fs)+windowk))),2)', EEG.chanlocs)

%% 5.4 Use a while or for loop to create a little movie of the topographical representation over time
%What do you see? Are you cool with this? Why not?


%% 6. Artifact rejection


figure
subplot(2,1,1)
plot(TimeX,squeeze(store_StimlockedEEG(:,1,:))')
xxd = ginput(1);
indbad1 = find(any(((squeeze(store_StimlockedEEG(:,1,:))) > xxd(1,2)),2));
idxgood = 1:size(triggers,2);
idxgood(indbad1) =  [];   
 subplot(2,1,2)
 plot(TimeX,squeeze(store_StimlockedEEG(idxgood,1,:))')
 
 %% 6.1 Exercise
 %if your computer is good enough (enough RAM, proper OpenGL graphics card,
 % proper OpenGL setup), try to plot all Channels (e.g. use a for loop)
 % and determine bad epochs based on all Channels
 
 
%% 6.1 Exercise update trialinfo (info2) and trialmatrixes accordingly
% idxbad2 = cleanscript(squeeze(store_StimlockedEEG(:,1,:)),Fs)
% if you want to have a bigger dataset and you are sure, your computer can
% handle that (8, better 16GB Ram)
%% 6.2 Load bigger  dataset for next exercise
% load data_ERP_small
% RTsx = (store_infoALL(:,4));
% [idxquick,c] = find(RTsx < median(RTsx) & median(RTsx) > .3 );
% [idxslow,h] = find(RTsx > median(RTsx) & median(RTsx) < 2 );
% This dataset has been epoched, and artifact trials are removed, however,
% data is not filtered with e.g. high-pass filter - so mean offsets can be
% expected.


%% 7. Statistics
% evaluate ERPs statistically
%do a simple test between mean amplitudes for one Channel
epochx = 500:600;
meanAmpslowCz = squeeze(mean(squeeze(store_StimlockedEEG(idxslow,65,epochx)),2));
meanAmpquickCz = squeeze(mean(squeeze(store_StimlockedEEG(idxquick,65,epochx)),2));
[h,p,ci,stats] = ttest2(meanAmpslowCz,meanAmpquickCz);

tstat = squeeze(stats.tstat)
p

%do a simple test between mean amplitudes for all Channel
meanAmpslow = squeeze(mean(store_StimlockedEEG(idxslow,:,epochx),3));
meanAmpquick = squeeze(mean(store_StimlockedEEG(idxquick,:,epochx),3));
[h,p,ci,stats] = ttest2(meanAmpslow,meanAmpquick);

tstat = squeeze(stats.tstat);
figure
subplot(2,1,1)
plot(tstat)
subplot(2,1,2)
plot(p)
ylim([0 0.05])

%% 7.1 Exercise: Significance levels & multiple comparisons
% what t-value would you deem statistically significant? 
% e.g. use the bonferooni correction for multiple comparisons to estimate a significance level 
% in case of bonferonni, you have to divide the critical p-value by the
% number of comparisons to yield your adjusted significance level
% say you use p = 0.05 and you compare mean amplitudes of all channel
% at 1 epoch: your new p-value is 
sig = 0.05;
numofcomparisons =67;
p_crit = sig/numofcomparisons


%% 7.2 Test the null hypothesis that at each timepoint (and each electrode)
%the difference between the two conditioons is 0 with an unknown SD
[h,p,ci,stats] = ttest2(squeeze((store_StimlockedEEG(idxslow,:,:))),squeeze((store_StimlockedEEG(idxquick,:,:))));

tstat = squeeze(stats.tstat);
figure;
subplot(3,1,1)
plot(TimeX,tstat(1,:))
axis tight
subplot(3,1,2)
plot(TimeX,squeeze(p(:,1,:)))
ylim([0 0.05])



%% 7.2 Non paremetric permutation testing
% construct statistical significance level with non-parametric permutation testing
% establish a t distribution on shuffled data

%% 7.21 Create t-distribution with shuffled data

 for tt=1:100               
  temp_store = [squeeze((store_StimlockedEEG(idxslow,:,:))) ;squeeze((store_StimlockedEEG(idxquick,:,:)))];
  % get length of matrix and permute indices, use shuffled indices to
  % construct matrices containing randomly chosen trials regardless of
  % trialtype, use those matrices to do a t-test, save the t-stats for each
  % run
  idx_length = size(temp_store,1);
  idx_rand = randperm(idx_length);
   
  quick_rand = temp_store((idx_rand(1:ceil(length(idx_rand)/2))),:,:);
  slow_rand = temp_store((idx_rand(ceil(length(idx_rand)/2+1):end)),:,:);
  [h,p,ci,stats] = ttest2(slow_rand,quick_rand);
% remember also to do a F-test for equal variance first, this ttest used here assumes equal variance in the 2 datasets  
  t_distribution(tt,:,:) = squeeze(stats.tstat);
  
 end
  
 %% 7.22 The probability distriubution 

 pd = fitdist(t_distribution(:,1,500),'Normal');
figure
x_values = -5:.1:5;
y = pdf(pd,x_values);
plot(x_values,y,'LineWidth',2)

 
SP=pd.mu-1.96*pd.sigma; %your point goes here 
 
SP2=pd.mu-2.577*pd.sigma; %your point goes here
line([SP SP],[0 0.1])
line([SP2 SP2],[0 0.1])

% 95% confidence interval (CI): 1.96*z, 99% CI: 2.577*z, 99.9% CI: 3.3*z
% depending on how many electrodes you are interested here it might be appropriate to
% introduce another correction for multiple comparisons by adjusting the
% confidence interval you use, by e.g. bonferonni correction.

 
%% 7.23 calculate confidence intervals at 95% for each electrode and each timepoint from the before estimated distribution
 % do that for each electrode and each timepoint seperatedly
  
  aa = size(t_distribution);
  for bb = 1:aa(2)
      for cc = 1:aa(3)
%       pd = fitdist(t_distribution(:,bb,cc),'Normal');
%      do_border(bb,cc,:) = pd.mu-1.96*pd.sigma;
%      up_border(bb,cc,:) = pd.mu+1.96*pd.sigma;
%      
     do_border2(bb,cc,:) = mean(t_distribution(:,bb,cc))-1.96*std(t_distribution(:,bb,cc));
     up_border2(bb,cc,:) = mean(t_distribution(:,bb,cc))+1.96*std(t_distribution(:,bb,cc));
     
%modelling the distribution and computing mean and standard deviation is
%slow, much quicker and same results without modelling
      end
  end
  
  
  
%% 7.3 Plot t-statisctics and significance level at p = 0.05 for one channel

figure;
plot(TimeX,tstat(1,:))
title(strcat('slow RT vs quick RT t-test at   ', Channelnames(1)))
hold on
plot(TimeX,do_border2(1,:),'g')
hold on
plot(TimeX,up_border2(1,:),'g')

%% 8. Lateralized Readiness Potential
% construct lateralized readiness potential LRP as [EEGright-EEEleft]@lefthand answers  + [EEGleft-EEGright]@righthand answers - Channels at motorcortex to usw
clearvars
%% Load dataset for next exercise
load data_ERP_small

idxleftanswer = find(store_infoALL(:,2) == 2);
idxrightanswer = find(store_infoALL(:,2) == 1)

storeleft = squeeze(store_ResponselockedEEGALL2(idxleftanswer,15,:)-store_ResponselockedEEGALL2(idxleftanswer,13,:));
storeright = squeeze(store_ResponselockedEEGALL2(idxrightanswer,15,:)-store_ResponselockedEEGALL2(idxrightanswer,13,:));
LRP = mean(storeleft)+mean(storeright);

TimeZ = (1/Fs:1/Fs:size(store_ResponselockedEEGALL2,3)/Fs)-2;
figure
plot(TimeZ,LRP)
%% 8.1  Exercise
% Read infos about different dataset in 6.2
% Why do you think the LRP doesn't look as expected?
% Try to implement a solution here, by filtering the trials from the used channels with a proper
% filter set and rejecting artifacts, you can use the code above for your
% solution

%% 8.2 Exercise

% You can now look at error trials, slow trials, quick
% trials, different conditions, for this u have to find the indices for the
% different conditions, seperated by answer hand - you have the indices
% already, to check for elements in both vectors of indices, you can use
% the ismember function.4RTsx = (store_infoALL(:,4));
% [idxquick,c] = find(RTsx < median(RTsx) & median(RTsx) > .3 );
% [idxslow,h] = find(RTsx > median(RTsx) & median(RTsx) < 2 );

% Use the ERP dataset containing over 500 trials
% Implement apropriate filter, do artifact rejection, and compare ERPs and LRPs from slow vs
% quick trials (with no error (column3 in infomatrix)) on an individual
% sensor level as well as on a topographical level and evaluate the
% differences statistically



%% Bonus - 9. Classification
% for those of you who have the a newer than Matlab 2015 version, try using a classifier to distinguish between two trialtypes
% use the timecourses of two different trialtypes as input for the
% classifier; for simplicity, use the ClassificationLearner APP from the
% Matlab APP bar and feed it with a matrix that contains all the single timecourses of
% both trialtypes and an indication (e.g. 1 & 2) to which class of trials a certain
% timecourse belongs (e.g. put a 1 or 2 in the matrix for each trial after
% all the samples and choose it as Response in the APP



%% 9.1 Exercise1: SVM Classifier and classification features
% Use a Support Vector Machine (SVM) Classifier and plot the beta values of
% the classifier in trainedClassifier.Beta which indicate which features
% the classifier uses to distinguish between classes; compare the beta
% values with the timecourse of a t-test (you have to normalize both
% vectors
 timeframe = Fs+50:Fs+Fs;
  temp_store = [squeeze((store_StimlockedEEG(idxslow,:,timeframe))) ;squeeze((store_StimlockedEEG(idxquick,:,timeframe)))];


%for now: only train classifier on one channel, every timepoint of every
%trial of both trialtypes we want to distinguish (labeled 1 and 2) is a
%feature for the classifier; in principle you can hand the classifier as
%many features as you want. usually before you train a classifier you may
%want to extract meaningful features that describe the data well and can
%possibly disinguish between the classes. there are approaches which automate
%that, google deep learning.

temp_storeXY = squeeze(temp_store(:,1,:)); 
size(squeeze((store_StimlockedEEG(idxslow,:,:))))
temp_storeXY(1:length(idxslow),size(temp_storeXY,2)+1) = 1;
temp_storeXY(length(idxslow)+1:end,size(temp_storeXY,2)) = 2;
classificationLearner

% select last row as response, train a linear SVM classifier,
% export it; you can export your classifier and classifier function


plot(trainedClassifier.Beta)

%% 9.2 Exercise 2: Accessing the significance of feature weights
% Try to construct a significance level for the beta weights with
% permutation testing as you have done that for the t-tests. (shuffle the
% column which contains the classes, retrain a SVM classifier, save the
% beta weights, do that n times. construct 95% confidence levels for the
% obtained distribution, plot significance levels and original beta weights
% in one plot.
%

%% 10. Toolboxes
% if you are interested in a more grphically guided approach try  brainstorm toolbox http://neuroimage.usc.edu/brainstorm/
% look at the tutorials on the webpage and do a ERP analysis on your
% own. Questions will be answered in the seminar. Other toolboxes you might
% try and probably need later in the Course are Fieldtrip:
% http://www.fieldtriptoolbox.org/ and EEGlab http://sccn.ucsd.edu/eeglab/


%% 13. Frequency analysis
% Introduction: Evoked vs induced activity
% Frequency decomposition of a signal, the Fourier transformation
% Bandpassfiltering theory, Power envelopes
% Introduction to wavelets in T-F-Analysis

%% 13.1 FFT on artifical data, play around with parameters


% create some noisy data
samplerate      = 4096;                     % Hz
duration        = 10;                       % seconds
time            = 1/samplerate:1/samplerate:duration;  % time vector
frequencies     = [0.5 1 2 5 7 10 12 25 50 100 150 200 202 250];
amplitudes      = ones(size(frequencies)).*0.75;
segmentLengths  = [512 1024 4096 8192 32768];

y = zeros(size(time));                      % combine sinusoids
for (i=1:length(frequencies))
    y = y + amplitudes(i)*sin(2*pi*frequencies(i)*time);
end
y = y + 2*randn(size(time));                % add some noise

% plot the trace
figure
plot(time,y)
title('artificial data')
xlabel('time [s]')
ylabel('intensity [arb. units]');

% A) Use the Fast Fourier transformation using the FFT function from Matlab to estimate the power spectrum
% http://de.mathworks.com/help/matlab/ref/fft.html

ps = abs(fft(y)); % the power spectrum
freq   = [0 (samplerate/length(y):samplerate/length(y):samplerate/2)]; % the frequency vector

figure
plot(ps)
xlabel('frequency [Hz]');
ylabel('power');

ps_ss =  ps(1:floor(length(ps)/2)+1); % single-sided spectrum (positive freqs only)

figure
xlabel('frequency [Hz]');
ylabel('power');
plot(freq,ps_ss)

%% 13.2 Use windows of different sizes to estimate power spectrum average

for i=1:length(segmentLengths)  % cycle through segment lengths
    freq               = [0 (samplerate/segmentLengths(i):samplerate/segmentLengths(i):samplerate/2)];
    noOfSamples		= length(y);
    noOfSegments	= floor(noOfSamples/segmentLengths(i));
    powers          = zeros(length(freq ),noOfSegments);    % some space for the resulting power spectra.
    
    for j=1:noOfSegments % create psd for each segment
        start	= (j-1)*segmentLengths(i)+1;
        ende	= start+segmentLengths(i)-1;
        segment	= y(start:ende);
        py      = abs(fft(segment,segmentLengths(i))).^2; % the power spectrum
        powers(:,j) = py(1:floor(length(py)/2)+1);        % single-sided spectrum (positive freqs only)
    end
    
    figure
    plot(freq ,mean(powers,2));
    title(sprintf('average power spectrum using segments of %d samples length. Duration of dataset: %d seconds @ %d Hz'...
        ,segmentLengths(i),duration,samplerate));
    xlabel('frequency [Hz]');
    ylabel('power');
end



%% 13.3 FFT on real data, whole trace


samplerate      = sampfreq; % Hz
duration        = length(data_filtered(1,:))/samplerate; %seconds
segmentLength   = 4096; %samples
time            = 1/samplerate:1/samplerate:duration;  % time vector
f               = [0 (samplerate/segmentLength:samplerate/segmentLength:samplerate/2)]; %frequency vector
noOfSamples		= length(data_filtered(1,:))
noOfSegments	= floor(noOfSamples/segmentLength);
windowx          = hann(segmentLength); % we build a hanning kernel to later convolute the segments to avoid edge effects

%power spectrum analysis
powers          = zeros(length(f),noOfSegments);    % some space for the resulting power spectra
for j=1:noOfSegments % create psd for each segment
    start	= (j-1)*segmentLength+1;
    ende	= start+segmentLength-1;
    segment	= data_filtered(1,start:ende)';
    segment = segment - mean(segment,1); % baseline correction
    pSegment= abs(fft(segment.*windowx,segmentLength)).^2; % the power spectrum
    powers(:,j) = pSegment(1:floor(length(pSegment)/2)+1);  % single-sided spectrum (positive freqs only)
end
FrequencyPower = mean(powers,2);

figure
% plot results
subplot(2,1,1)
plot(f,FrequencyPower);
% xlim([1 1000])
% ylim([100 100000])
title('power spectrum')
xlabel('frequency [Hz]')
ylabel('power')

%% 13.4 FFT on Epochs with the use of pwelch
% Pwelch is a Matlab function that does
% return the power spectral density (PSD) estimate, using Welch's
% overlapped segment averaging method (does the same as done manually
% above)
% http://de.mathworks.com/help/signal/ref/pwelch.html
% we will use only on Channel here, try to compare the midline-frontal
% theta in different trial types

nn = size(store_StimlockedEEG,1)
nfft = 512;
for i = 1:nn(1)
    
    [powerstoreSI(i,:), F] =  pwelch(store_StimlockedEEG(i,1,:),hanning(nfft),nfft/2,nfft,sampfreq); % estimation of PSD using pwelch function
      
end

nn = size(SC_epochs)
nfft = 512;
for i = 1:nn(1)
    
    [powerstoreSC(i,:), F] =  pwelch(SC_epochs(i,1,:),hanning(nfft),nfft/2,nfft,sampfreq);
      
end

figure
plot(F,mean(powerstoreSI),'b')
hold on
plot(F,mean(powerstoreSC),'g')
title('PSD')
xlabel('frequency [Hz]')
ylabel('power')

%% 13.5 Exercise1: Do statistical testing on the power spectrum estimates
% Use the logic provided in the ERP session, use power spectrum estimates
% for each trial and do a t-test using all trials from each condition
% shuffle the labels of the data, estimate a t-distribution and construct
% confidence intervals



%% 14. Time-frequency analysis
%% 14.1 Short-time-Fourier Transformation (STFT) on epochs
% using spectrogram function from
% Matlabhttp://de.mathworks.com/help/signal/ref/spectrogram.html on each
% epoch, then do average over epochs

nn = size(SC_epochs)
nfft = 512;
% calculate STFT for each segment
for i = 1:nn(1)
    
   segmentx = squeeze(squeeze(SC_epochs(i,1,:)));
   [S,F,T,P] = spectrogram(segmentx,gausswin(nfft),nfft-1,nfft,sampfreq);
   S1 = abs(S);
   
%    ff = size(S1);
%    for n = 1:ff(2)
%        S1(n,:) = S1(n,:)-mean(S1(n,1:100));
%    end
   
   Sstore(i,:,:) = (S1);
   Pstore(i,:,:) = P;
end

Smean = squeeze(mean(Sstore));
Pmean = squeeze(mean(Pstore));

figure
set(gcf,'renderer','zbuffer') 
surf(T,F,Pmean,'edgecolor','none', 'facecolor', 'interp'); 
view(2);
axis tight

figure
set(gcf,'renderer','zbuffer') 
surf(T,F,Smean,'edgecolor','none', 'facecolor', 'interp'); 
view(2);
axis tight
%% 14.2 Exercise. Baseline Correction
% Do a baseline correction to see the Event-related Spectral Pertubations


%% 15. Bandpassfiltering & power envelope
% in contrary to the above method, this time we first transform the whole
% signal, construct its power envelope and extract epochs afterwards
% together with averaging

fcutlow = 1:4:sampfreq/2-1;
fcuthigh = (1:4:sampfreq/2-3)+4;
order = 4;
fcutlow(end) = []


for tt=1:length(fcutlow)
[b,a]    = butter(order,[fcutlow(tt),fcuthigh(tt)]/(sampfreq/2), 'bandpass');
x        = filter(b,a,data_filtered(1,:));
data_bpfiltered(tt,:) = x;
envelope(tt,:) = abs(hilbert(x));
end

% check your results
figure
plot(data_bpfiltered(2,2000:3000))
hold on
plot(envelope(2,2000:3000),'g')
hold on
plot(data_bpfiltered(3,2000:3000),'r')

% what is plotted here. does it make sense?



ll = 1;
for i = 1:length(SC_trials_ultraclean)
    %cut out one epoch after the other
temp = envelope(:,SC_trials_ultraclean(i)-windowepochs:SC_trials_ultraclean(i)+windowepochs);
%     if max(temp(1,:)) < threshold && min(temp(1,:))> -threshold %set another artifact rejection criterion here, reject all trials containing amplitudes over/under treshold
% for l=1:electrodescount
%     %do baseline correction for all epochs for each electrode seperatedly
% temp(l,:) = temp(l,:)-mean(temp(l,1:baselinewindow));

% end

SC_epochs_BPF(ll,:,:) = temp;

ll = ll+1;
   
clearvars temp

end

SmeanBPF = squeeze(mean(SC_epochs_BPF));
figure
set(gcf,'renderer','zbuffer') 
surf(1:1:751,fcutlow,SmeanBPF,'edgecolor','none', 'facecolor', 'interp');
view(2);
axis tight

% compare results, what do you recognize?
%% 15.1 Exercise. Take a look at different Channels by looping the analysis across channels

% 6. Wavelets
wavemenu

% try out the DWT or CWT to compute time-frequency transformation; to compute frequency resolution from scales, use: http://de.mathworks.com/help/wavelet/ref/scal2frq.html



