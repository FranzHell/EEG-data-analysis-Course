function [idxd] = cleanscript(datax,Fs)
times = 1/Fs:1/Fs:size(datax,2)/Fs;
xx=1;
for i = 1:size(datax,1)
             
datax = squeeze(datax);
     
      tempd = squeeze(datax(i,:));          
          tempd = tempd - mean(tempd);
   
     %% make larger wavelet, cut ends
   tempod = morlet_transform(tempd, times,[1,3,8,13,30,50]);                 
storeLFPwave(xx,:,:) = squeeze(tempod(1,100:end-100,:));

xx = xx+1;

         end
             xx = 1;
          
      
         
      
      
         sync_channel = 1;
         figure
         yy=1;
         while sync_channel < 7                                                          % sync channel ist hier die Grafik auf die geklickt wird, 9 ist weiter
%              ind = find(step_parameter(:,1,5) == side);
             clf
             for chan=1:6
                 ax(chan)= subplot(3,2,chan);
               sp = (squeeze(storeLFPwave(:,:,chan)));
                 plot(sp')
%                  text(0.5, 0.92,freq_txt{chan},'Units','normalized','HorizontalAlignment','center','FontSize',8)
             end
             axes('position', [0 0 1 1],'visible', 'off')
%              if side == 1, tx ='Schritt rechtes Bein'; else tx ='Schritt linkes Bein'; end
%              text(0.5, 0.97,tx,'HorizontalAlignment','center','FontSize',14)
             text(0.5, 0.02,'Artefaktschwelle markieren oder auf Weiter klicken','HorizontalAlignment','center','FontSize',14)


             ax(7) = axes('position', [0.925 0.17 0.05 0.04],'visible', 'off');                                         % wenn auf diese Achse geklickt wird gehts weiter
             text(0.5, 0.5,'weiter', 'HorizontalAlignment','Center','FontSize',16,'FontWeight','Bold','col','r')

             [~,t]=ginput(1);
             sync_channel=find(get(gcf,'CurrentAxes') == ax);
             if sync_channel < 7
                ind2 = find(any(((squeeze(storeLFPwave(:,:,sync_channel))) > t),2));
%                 ind2=ind(ind2);
                
                indexesdelete{yy} = ind2';
                yy=yy+1;
                storeLFPwave(ind2,:,:,:) = []; 

             end
         end
       idxd=cell2mat(indexesdelete);
       end