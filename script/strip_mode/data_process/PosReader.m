classdef PosReader
  %POSREADER Read POS data

  properties (SetAccess = private)
    fileName
    timestamp
    north
    east
    height
  end

  methods
    function self = PosReader(fileName)
      %POSREADER Construct an instance of PosReader
      %   fileName: POS data file path

      % read file
      self.fileName = fileName;
      f_pos = fopen(fileName, 'r');
      if f_pos == -1
        fprintf(2, 'Can not open file: %s\n', fileName);
        return
      end
      pos_data = fread(f_pos, Inf, '*double');
      fclose(f_pos);

      % organize data
      pos_data = reshape(pos_data, [17, numel(pos_data)/17]);

      % timestamp
      day_offset = floor(pos_data(1,1)/24/3600)*24*3600;
      self.timestamp = pos_data(1,:) - day_offset;

      % coordinates
      lat = pos_data(2,:);
      lng = pos_data(3,:);
      h = pos_data(4,:);
      [self.east, self.north, self.height] = gps2xyz(lng, lat, h);
    end

    function [forward, right, down] = getCoords(self, time)
      %GETCOORDS Get coordinates corresponding to the POS time in echo data
      
      % find time slice
      time = time2sec(time);
      slcb = find(self.timestamp>=time(1), 1);
      if isempty(slcb)
        slcb = 1;
      else
        slcb = max(1, slcb - 1);
      end
      slce = find(self.timestamp>time(end), 1);
      if isempty(slce)
        slce = numel(self.timestamp);
      else
        slce = min(numel(self.timestamp), slce);
      end
      
      % pre-processing raw coordinates data
      local_timestamp = self.timestamp(slcb:slce);
      local_north = self.north(slcb:slce);
      local_east = self.east(slcb:slce);
      local_height = self.height(slcb:slce);
      % pad_size = 2;
      % smooth_filter = ones([1, 2*pad_size+1]) ./ (2*pad_size+1);
      % slcb_pad = max(1, slcb - pad_size);
      % slce_pad = min(numel(self.timestamp), slce + pad_size);
      % local_timestamp = self.timestamp(slcb_pad + pad_size:slce_pad - pad_size);
      % local_north = conv(self.north(slcb_pad:slce_pad), smooth_filter, 'valid');
      % local_east = conv(self.east(slcb_pad:slce_pad), smooth_filter, 'valid');
      % local_height = conv(self.height(slcb_pad:slce_pad), smooth_filter, 'valid');

      % find forward direction (sea level plane)
      plane_coords = [local_north; local_east];
      % heading = plane_coords(:,end) - plane_coords(:,1);
      heading_line = polyfit(plane_coords(1,:), plane_coords(2,:), 1);
      heading = sign(plane_coords(1, end)-plane_coords(1, 1)) .* ...
        [1; polyval(heading_line, 1) - polyval(heading_line, 0)];
      heading = heading ./ realsqrt(sum(heading.^2));  % normalize heading vector

      % calculate coordinates
      forward = heading.' * plane_coords;
      right = [heading(2), -heading(1)] * plane_coords;
      down = -local_height;
      
      % centering
      forward = forward - mean(forward);
      right = right - mean(right);
%       down = down - mean(down);

      % interpolating
      forward = interp1(local_timestamp, forward, time, 'linear', 'extrap');
      right = interp1(local_timestamp, right, time, 'linear', 'extrap');
      down = interp1(local_timestamp, down, time, 'linear', 'extrap');
      % forward = interp1(local_timestamp, forward, time, 'spline', 'extrap');
      % right = interp1(local_timestamp, right, time, 'spline', 'extrap');
      % down = interp1(local_timestamp, down, time, 'spline', 'extrap');

    end
  end
end

%% helper functions

function [seconds] = time2sec(time)
%TIME2SEC convert echo recorded time to pos seconds

% split hhmmss digits
time = double(time);
hours = floor(time/1e4);
minutes = floor((time-hours*1e4)/1e2);
seconds = time-hours*1e4-minutes*1e2;

% add hours and minutes
timezone = 8;
seconds = seconds + (hours - timezone)*3600 + minutes*60;  % time zone conversion

% add fractional part
% % FIXME result is not accurate
% % sec_change = find(diff(seconds)) + 1;
% % poly = polyfit(sec_change(1):sec_change(end), ...
% %   seconds(sec_change(1):sec_change(end)), 1);
% poly = polyfit(1:numel(seconds), seconds, 1);
% seconds_new = polyval(poly, 1:numel(seconds));
% seconds_new = reshape(seconds_new, size(seconds));
% % sec_diff = -min(seconds_new - seconds);
% sec_diff = 0.5;
% seconds_new = seconds_new + sec_diff;
sec_change_idx = find(diff(seconds)~=0) + 1;
poly = polyfit(sec_change_idx, seconds(sec_change_idx), 1);
seconds_new = polyval(poly, 1:numel(seconds));

seconds = seconds_new;

end

function [deast, dnorth, dalti] = gps2xyz(gpslong, gpslat, gpsalti)
%功能：已知初始经纬高和目标经纬高，计算目标点在初始点东向多少米，北向多少米

%数据输入
platform_lon = gpslong(1);        %初始经度
platform_lat = gpslat(1);         %初始纬度
platform_alt = gpsalti(1);                %初始高度
target_lon = gpslong;          %目标经度
target_lat = gpslat;           %目标纬度
target_alt = gpsalti;                  %目标高度

%中间参数
Re=6378137;           %地球半长轴，单位：米, WGS-84
e=0.003352810929627;  %(1/298.2572)         //地球扁率 f=（Re-Rp)/Re

Rs = sin(platform_lat);
Rm = Re * (1 - 2 * e + 3 * e * Rs * Rs);
Rn = Re * (1 + e * Rs * Rs);


%初始位置与目标位置的经度差值，换算成东向距离，单位m
deast = (target_lon - platform_lon)*(Rn+platform_alt)*cos(platform_lat);

%初始位置与目标位置的纬度差值，换算成北向距离，单位m
dnorth = (target_lat - platform_lat)*(Rm+platform_alt);

dalti = target_alt;

end
