%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;clear;close all;
%代码格式化命令：MBeautify.formatCurrentEditorPage()

%% 卫星轨道参数

H = 755e3; %卫星轨道高度
% 卫星轨道速度Vr计算
EarthMass = 6e24; %地球质量(kg)
EarthRadius = 6.37e6; %地球半径6371km
Gravitational = 6.67e-11; %万有引力常量
% 姿态参数
phi = 20 * pi / 180; % 俯仰角+20°
incidence = 20.5 * pi / 180; % 入射角
%计算等效雷达速度(卫星做圆周运动的线速度)
Vr = sqrt(Gravitational*EarthMass/(EarthRadius + H)); %第一宇宙速度

%景中心斜距R_eta_c和最近斜距R0，斜视角theta_rc由轨道高度、俯仰角、入射角计算得出
R_eta_c = H / cos(incidence); %景中心斜距
R0 = H / cos(phi);
theta_r_c = acos(R0/R_eta_c); %斜视角，单位为弧度;斜视角为4.6°

%% 信号参数设置

%   电磁波参数
c = 3e+8; % 光速
Vs = Vr; % 卫星平台速度
Vg = Vr; % 波束扫描速度


La = 15; %方位向天线长度->椭圆的长轴!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Lr=1.5;%距离向天线尺寸——>椭圆的短轴!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



f0 = 5.4e+9; % 雷达工作频率
lambda = c / f0; %电磁波波长


%  距离向信号参数
Tr = 40e-6; % 发射脉冲时宽
Br = 2.8 * 6e6; % 距离向信号带宽
Kr = Br / Tr; % 距离向调频率
alpha_os_r = 1.2; % 距离过采样率
Nrg = 2500; % 距离线采样点数     
Fr = alpha_os_r * Br; % 距离向采样率


%  方位向信号参数
alpha_os_a = 1.7; % 方位过采样率(高过采样率避免鬼影目标)
Naz = 1600; % 距离线数  
delta_f_dop = 2 * 0.886 * Vr * (cos(theta_r_c)) / La; % 多普勒带宽
Fa = alpha_os_a * delta_f_dop; % 方位向采样率
Ta= 0.886 * lambda * R_eta_c / (La * Vg * cos(theta_r_c)); %目标照射时间

%  景中心点(原点)的参数
time_eta_c = -R_eta_c * sin(theta_r_c) / Vr; % 波束中心穿越时刻
f_eta_c = 2 * Vr * sin(theta_r_c) / lambda; % 多普勒中心频率

%  合成孔径参数
rho_r = c / (2 * Fr); % 距离向分辨率
rho_a = Vr/Fa; % 距离向分辨率|La / 2
theta_bw = 0.886 * lambda / Lr; % 方位向3dB波束宽度
theta_syn = Vs / Vg * theta_bw; % 合成角宽度(斜面上的合成角)
Ls = R_eta_c * theta_syn; % 合成孔径长度

%% 时间轴参数
Trg = Nrg / Fr;Taz = Naz / Fa;
%距离向/方位向采样时间间隔
Gap_t_tau = 1 / Fr;Gap_t_eta = 1 / Fa;
%距离向/方位向采样频率间隔
Gap_f_tau = Fr / Nrg;Gap_f_eta = Fa / Naz;
%  时间轴变量
time_tau_r = 2 * R_eta_c / c + (-Trg / 2:Gap_t_tau:Trg / 2 - Gap_t_tau); % 距离时间变量
time_eta_a = time_eta_c + (-Taz / 2:Gap_t_eta:Taz / 2 - Gap_t_eta); % 方位时间变量
%  随着距离向时间变化的最近斜距；c/2是因为距离向上一个时间包含两次电磁波来回
R0_tau_r = (time_tau_r * c / 2) * cos(theta_r_c);
Ext_R0_tau_r = repmat(R0_tau_r, Naz, 1); %扩展R0，用于计算变量Ka
%  频率变量
f_tau = (-Fr / 2:Gap_f_tau:Fr / 2 - Gap_f_tau); % 距离频率变量
f_tau=f_tau-(round(f_tau/Fr))/Fr;%混叠方程
f_eta = f_eta_c + (-Fa / 2:Gap_f_eta:Fa / 2 - Gap_f_eta); % 方位频率变量
f_eta=f_eta-(round(f_eta/Fa))/Fa;

%  时间轴
[Ext_time_tau_r, Ext_time_eta_a] = meshgrid(time_tau_r, time_eta_a); % 设置距离时域-方位时域二维网络坐标
%  频率轴
[Ext_f_tau, Ext_f_eta] = meshgrid(f_tau, f_eta); % 设置频率时域-方位频域二维网络坐标

%% 点目标(三个)坐标设置
%  设置目标点相对于景中心之间的距离

xA = 0;yA = 0; %A=(0,0)
xB = xA + 500;yB = xA + 500; %（500,500）

xC = H*tan(phi+theta_bw/2)-R0*sin(phi);%计算的C点距离向坐标
yC = xA + 500;
Position_x_r = [xA, xB, xC];
Position_y_a = [yA, yB, yC]; %点目标的坐标矩阵表示

%% 生成回波S_echo
Target_num = 3; %目标数量
S_echo = zeros(Naz, Nrg);
for i = 1:Target_num
    R0_Target = sqrt((R0 * sin(phi) + Position_x_r(i)) ^2+H^2); %对每个目标计算瞬时斜距
    
    time_eta_c_Target = (Position_y_a(i)-R0_Target * tan(theta_r_c)) / Vr; %波束中心穿越时刻

    %  计算目标点的瞬时斜距
    R_eta = sqrt((R0_Target)^2+(Vr^2)*((Ext_time_eta_a - Position_y_a(i) / Vr).^2));
    %R_eta=sqrt(H^2+(Position_x_r(i)-(-R0*sin(phi)))^2+(Position_y_a(i)-Ext_time_eta_a*Vr).^2);
    %  距离向包络
    Wr = (abs(Ext_time_tau_r-2*R_eta/c) <= Tr / 2);
    %  方位向包络
    %Wa = sinc((La*atan(Vg*(Ext_time_eta_a - time_eta_c_Target)./R0_Target)/lambda).^2);
    Wa = (La*atan(Vg*(Ext_time_eta_a - time_eta_c_Target)./(R0 * sin(phi) + Position_x_r(i))/lambda).^2)<=Ta/2;
    %Wa = abs(Ext_time_eta_a-time_eta_c_Target) <= Ta / 2;
    %   相位
    Phase = exp(-1j*4*pi*f0*R_eta/c) .* exp(+1j*pi*Kr*(Ext_time_tau_r - 2 * R_eta / c).^2);
    %  接收信号叠加
    S_echo_Target = Wr .* Wa .* Phase;
    S_echo = S_echo + S_echo_Target;
    %%%%%%%%%%%%%%格式化输出%%%%%%%%%%%%%%%%%%
    R_eta_c_Target=R0_Target/cos(theta_r_c);
    fprintf("当前目标:%d,坐标:(%.2f,%.2f),\n最近斜距R0=%.2f,景中心斜距R_eta_c=%.2f,波束中心穿越时刻=%.4f\n", i, Position_x_r(i), Position_y_a(i), R0_Target, R_eta_c_Target,time_eta_c_Target)
    Time_tau_Point =  round(((2*R0_Target)/(c*cos(theta_r_c))-time_tau_r(1))/Gap_t_tau);%目标的距离向标准坐标(距离门)
    Time_eta_Point = Naz / 2 + (Position_y_a(i) / Vr) /Gap_t_eta;
    fprintf("徙动校正前,点数坐标应为%d行(方),%d列(距)\n\n",round(Time_eta_Point),Time_tau_Point);
end
S_echo = S_echo .* exp(-2j*pi*f_eta_c*Ext_time_eta_a); %校正时间轴

%% 距离压缩

Hf = (abs(Ext_f_tau) <= Br / 2) .* exp(+1j*pi*Ext_f_tau.^2/Kr);%滤波器
%  匹配滤波
S1_ftau_eta = fftshift(fft(fftshift(S_echo, 2), Nrg, 2), 2);
S1_ftau_eta = S1_ftau_eta .* Hf;
S1_tau_eta = fftshift(ifft(fftshift(S1_ftau_eta, 2), Nrg, 2), 2);

%% 方位向傅里叶变换
S2_tau_feta = fftshift(fft(fftshift(S1_tau_eta, 1), Naz, 1), 1);

%% 距离徙动校正RCMC:采用相位补偿法

%虽然Ka是随着R0变化的，但是在相位补偿时需要假设R0是不变的
delta_R = (((lambda * Ext_f_eta).^2) .* R0) ./ (8 * (Vr^2)); %距离徙动表达式
G_rcmc = exp((+4j * pi .* Ext_f_tau .* delta_R)./c); %补偿相位
S3_ftau_feta = fftshift(fft(fftshift(S2_tau_feta, 2), Nrg, 2), 2); %在方位向傅里叶变换的基础上进行距离向傅里叶变换

S3_ftau_feta = S3_ftau_feta .* G_rcmc; %与补偿相位相乘
S3_tau_feta_RCMC = fftshift(ifft(fftshift(S3_ftau_feta, 2), Nrg, 2), 2); %距离向傅里叶逆变换

%距离徙动校正结束

%% 方位压缩
%  根据变化的R0计算出相应的Ka矩阵(距离向变化，方位向不变)
Ka = 2 * Vr^2 * cos(theta_r_c)^2 ./ (lambda * Ext_R0_tau_r);
%  方位向匹配滤波器
Haz = exp(-1j*pi*Ext_f_eta.^2./Ka);
Offset = exp(-1j*2*pi*Ext_f_eta.*time_eta_c);%偏移滤波器，将原点搬移到Naz/2的位置，校准坐标
%  匹配滤波
S4_tau_feta = S3_tau_feta_RCMC .* Haz.*Offset;
S4_tau_eta = fftshift(ifft(fftshift(S4_tau_feta, 1), Naz, 1), 1);

%% 目标的升采样切片
%采用二维频域补零的方式进行升采样操作;升采样为了提高目标的切片细节丰富程度，便于观察
CutResolution = 32; %切片尺寸
Profile_Position = [901, 1570]; %切片的中心点位置
fprintf("\n点目标(C点)坐标对应的距离门到雷达距离为:%.2f\n",(time_tau_r(Profile_Position(2)))*(c/2))

%切片的升采样倍数：10*CutResolution；也就是在二维频域补多少个零
S5_tau_eta_Cut = S4_tau_eta(Profile_Position(1)-CutResolution/2:Profile_Position(1)+CutResolution/2, Profile_Position(2)-CutResolution/2:Profile_Position(2)+CutResolution/2);%切片

[S_zero_fill]=fft_zero_fill(S5_tau_eta_Cut,10*CutResolution);%补零函数

%由于斜视角的存在，在对升采样切片进行剖面分析前，将方位向和距离向都通过角度旋转校正到便于剖面的方向上
S5_tau_eta_Cut_UP_Azi=imrotate(S_zero_fill,rad2deg(theta_r_c),'bilinear', 'crop');%方位向切片无需角度校正
S5_tau_eta_Cut_UP_Ran = imrotate(S_zero_fill, rad2deg(theta_r_c), 'bilinear', 'crop'); %角度校正
%求解升采样切片包络的abs最大值坐标，用于下面的剖面
[UP_Profile_Position_Ran,UP_Profile_Position_Azi] = find(max(max(abs(S5_tau_eta_Cut_UP_Ran)))==abs(S5_tau_eta_Cut_UP_Ran));

%幅度db化+搬移峰值至0dB
%基于上面的升采样插值结果->获得剖面
Abs_S5_Azi = abs(S5_tau_eta_Cut_UP_Azi(:, UP_Profile_Position_Azi)); %方位向剖面
Abs_S5_Azi = Abs_S5_Azi / max(Abs_S5_Azi); %移动峰值点
Abs_S5_Ran = abs(S5_tau_eta_Cut_UP_Ran(UP_Profile_Position_Ran, :)); %距离向剖面
Abs_S5_Ran = Abs_S5_Ran / max(Abs_S5_Ran);

%% 回波图可视化

figure('name', "回波可视化")
subplot(2, 2, 1);
imagesc(real(S_echo));
title('原始仿真信号实部');
xlabel('距离向时间 \tau');
ylabel('方位向时间 \eta');

subplot(2, 2, 2);
imagesc(imag(S_echo));
title('原始仿真信号虚部');
xlabel('距离向时间 \tau');
ylabel('方位向时间 \eta');


subplot(2, 2, 3);
imagesc(abs(S_echo));
title('原始仿真信号幅度');
xlabel('距离向时间 \tau');
ylabel('方位向时间 \eta');

subplot(2, 2, 4);
imagesc(angle(S_echo));
title('原始仿真信号相位');
xlabel('距离向时间 \tau');
ylabel('方位向时间 \eta');

%% 距离压缩可视化

%距离频谱可视化
figure('name', "回波频谱可视化")
subplot(2, 2, 1);
imagesc(real(S1_ftau_eta));
title('回波距离向实部');
xlabel('距离向频谱点数f_\tau');
ylabel('方位向时间 \eta');

subplot(2, 2, 2);
imagesc(imag(S1_ftau_eta));
title('回波距离向虚部');
xlabel('距离向频谱点数f_\tau');
ylabel('方位向时间 \eta');

subplot(2, 2, 3);
imagesc(abs(S1_ftau_eta));
title('回波距离向频谱幅度');
xlabel('距离向频谱点数f_\tau');
ylabel('方位向时间 \eta');

subplot(2, 2, 4);
imagesc(angle(S1_ftau_eta));
title('回波距离向相位');
xlabel('距离向频谱点数f_\tau');
ylabel('方位向时间 \eta');

%距离压缩结果
figure('name', "距离压缩时域结果")
subplot(1, 2, 1);
imagesc(real(S1_tau_eta));
title('实部');
xlabel('距离向时间 \tau');
ylabel('方位向时间 \eta');

subplot(1, 2, 2);
imagesc(abs(S1_tau_eta));
title('幅度');
xlabel('距离向时间 \tau');
ylabel('方位向时间 \eta');

%% 方位向傅里叶变换可视化

figure('name', "方位向傅里叶变换结果")
subplot(1, 2, 1);
imagesc(real(S2_tau_feta));
title('实部');
xlabel('距离向时间\tau');
ylabel('方位向频谱点数f_\eta');

subplot(1, 2, 2);
imagesc(abs(S2_tau_feta));
title('幅度');
xlabel('距离向时间\tau');
ylabel('方位向频谱点数f_\eta');

%% 距离徙动校正可视化

figure('name', "距离徙动校正结果")
subplot(1, 2, 1);
imagesc(real(S3_tau_feta_RCMC));
title('实部');
xlabel('距离向时间\tau');
ylabel('方位向频率点数 f_\eta');

subplot(1, 2, 2);
imagesc(abs(S3_tau_feta_RCMC));
title('幅度');
xlabel('距离向时间\tau');
ylabel('方位向频率点数 f_\eta');

%% 回波成像
figure('name', "点目标成像结果")
subplot(1, 2, 1);
imagesc(abs(S4_tau_eta));
title('幅度');
xlabel('距离向时间\tau');
ylabel('方位向时间 \eta');

subplot(1, 2, 2);
imagesc(abs(S5_tau_eta_Cut)); %切片
title('点目标C的32×32切片(角度校正前)');
xlabel('距离向时间\tau');
ylabel('方位向时间\eta');


%% 升采样成像
figure('name', "升采样成像结果")
subplot(1, 2, 1);
imagesc(abs(S5_tau_eta_Cut_UP_Ran));
title('幅度');
xlabel('距离向时间\tau');
ylabel('方位向时间 \eta');

subplot(1, 2, 2);

%contour((1:Nrg),(1:Naz),abs(S5_tau_eta_Cut_UP_Ran),20);
contour(abs(S5_tau_eta_Cut_UP_Ran), 30);
title('升采样轮廓图');
xlabel('距离向时间\tau');
ylabel('方位向时间\eta');

%% 切片包络幅度(dB)剖面
figure('name', "目标C的距离和方位幅度包络切片(插值后)")

subplot(1, 2, 1);
plot(mag2db(Abs_S5_Ran));
title('目标A距离向包络剖面(有角度校正)');
xlabel('距离向时间\tau');
ylabel('包络剖面幅度(dB)');
yticks(-50:2:50); % 设置刻度间隔为0.1
%画一条-13dB的参考线
x = linspace(0, 500, 100); % 生成 x 坐标的数据
y = repmat(-13,100); % 生成对应的 y 坐标的数据，全都是-13
hold on;
line(x, y,'Color','red'); % 绘制直线

subplot(1, 2, 2);
plot(mag2db(Abs_S5_Azi));
title('目标A方位向包络剖面(无角度校正)');
xlabel('方位向时间\eta');
ylabel('包络剖面幅度(dB)');
yticks(-50:2:50); % 设置刻度间隔为0.1

%画一条-13dB的参考线
x = linspace(0, 500, 100); % 生成 x 坐标的数据
y = repmat(-13,100); % 生成对应的 y 坐标的数据，全都是-13
hold on;
line(x, y,'Color','red'); % 绘制直线

%% 切片相位剖面
figure('name', "目标C的距离和方位幅度包络切片(插值后)")
subplot(1, 2, 1);
plot(angle(S5_tau_eta_Cut_UP_Ran(UP_Profile_Position_Ran, :)));
title('目标A距离向相位剖面(有角度校正)');
xlabel('距离向时间\tau');
ylabel('包络剖面幅度(dB)');


subplot(1, 2, 2);
plot(angle(S5_tau_eta_Cut_UP_Azi(:, UP_Profile_Position_Azi)));
title('目标A方位向相位剖面(无角度校正)');
xlabel('方位向时间\eta');
ylabel('包络剖面幅度(dB)');
%% 手动补零


function [S_FFT]=fft_zero_fill(x,nums)%进行补零；补零前一定要观察二维频谱，避免将补零的数组插到有能量的(黄色)区域，破坏原本的频谱！
 S_FFT=fft(x,[],1);
 S_FFT=fft(S_FFT,[],2);%二维频谱；这里不使用fftshift，避免频谱被搬移到中心
figure('name',"二维频域补零前");
imagesc(abs(S_FFT));%将二维频谱可视化，确定补零的范围(中点)

Y_Insert=zeros(nums,33);%在Y方向插入(方位向)

S_FFT=[S_FFT(1:21,:);Y_Insert;S_FFT(22:33,:)];%插入补零的数组;肉眼观察二维频谱的补零数组插入位置！！!
X_Insert=zeros(size(S_FFT,1),nums);%在X方向插入

S_FFT=[S_FFT(:,1:21),X_Insert,S_FFT(:,22:33)];%插入补零的数组;肉眼观察二维频谱的补零数组插入位置！！!
figure('name',"二维频域补零后");
imagesc(abs(S_FFT));%将二维频谱可视化，确定补零的范围(中点)
S_FFT=(ifft(ifft(S_FFT,[],1),[],2));%反变换，回到二维时域

% 
% figure;
% imagesc(abs(ifft(ifft(S_FFT,[],1),[],2)))
end
