%拆分成两个文件 PP成本算150每平方 RG雨水花园算600  brc算500  RG BRC PP顺序计算
%由于约束条件复杂，直接把约束放在了这里面

% function [cb,yl]=paper1_problem1(mian_ji)
function CB=paper1_problemCB2(mian_ji)  %输入为 3*4=12
%成本和溢流量  xij来表示，LID类型与子汇水数量 约束条件为列之和不超过总的
%成本不需要计算直接算钱
    zhsqy_number=length(mian_ji)./3;
    CB_before=reshape(mian_ji.*10000,3,zhsqy_number)  ;%转换后是竖着的 


% fileID = fopen('F:\博士研究生\1.inp','r+');  %打开一个文件                  %以可读写的方式打开待修改的文件
% i=0;
% newline={};  %不加这句话，直接人都麻了 调试个BUG调了半天，就是这里出问题了，艹TMMMMMMPPPPPP！！！！
% while ~feof(fileID)   %未到文件末尾，便循环
%     tline=fgetl(fileID);                              %逐行读取原始文件
%     i=i+1;
%     newline{i} = tline;                               %创建新对象接受原始文件每行数据
%     %下面可以不用if 直接将多少行设置好就行
% 
% end
% lia3=contains(newline,'SUBCATCHMENTS');
% lia33=find(lia3,1);%查到位置所在行
% lia4=contains(newline,'SUBAREAS');
% lia44=find(lia4,1);%查到位置所在行
% % lia33+3----lia44-2为子汇水区域所在位置
% zhsqy=[lia33+3,lia44-2];
% zhsqy_number=lia44-2-lia33-3+1;%数量行包含自己，要加一
% mj=cell(0);
% mjjz=zeros(0);%子汇水区域面积矩阵初始化
% for isi=1:zhsqy_number
%     mj{isi}=newline{lia33+3-1+isi}(47:58);% 警告！！！此处根据具体面积所在位置进行更改 请看文本文档中的位置
%     mj{isi}=strip(mj{isi},' ');%去除空格，只留文字
%     mjjz(isi)=str2double(mj{isi});
% end
% Ss=sum(reshape(mian_ji,3,zhsqy_number));
% % for iii=1:length(mjjz)
% % S(iii)=
% % end
% S=Ss-mjjz;
% if S>0 
%     CB=10^9;
% else

    CB=sum([600 500 150]*CB_before);
    %%

%% 计算结束 开始下一步，提取文件
%提取out文件

end
