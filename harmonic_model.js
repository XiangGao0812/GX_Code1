//线性光谱混合分解
var LSMA = function (image){
  var SL  = [0.1001822,0.1601154,0.238036,0.2580122,0.2755076,0.2842358,0.2863682,0.3096766,0.3824306,0.336347];
  var GV  = [0.035406333,0.076643667,0.048356,0.111966,0.423144667,0.534067333,0.500459,0.598840667,0.262011333,0.117481667];
  var DA  = [0.018315,0.042391,0.027983,0.019284,0.008875,0.012169,0.011898,0.009591,0.008432,0.00747];
 
  var endmembers = [SL,GV,DA];
  var renameBands = ['SL','GV','DA'];
  var bands = ['B2', 'B3','B4','B5','B6','B7','B8','B8A','B11','B12'];
  
  var em =ee.Image(ee.Array(endmembers).transpose());
  var unmixed = function(img) {
    var ks = img.select(bands);
    var col = ks.unmix(endmembers,true,true).rename(renameBands);
    /*
    var colarray = col.toArray().toArray(1);// RMSE
    var REcon = em.matrixMultiply(colarray)
    .arrayProject([0])
    .arrayFlatten([['B2', 'B3','B4','B5','B6','B7','B8','B8A','B11','B12']]);
    var ks1 = ks.subtract(REcon);//减
    var rmse = ks1.expression(
    'sqrt((b2*b2+b3*b3+b4*b4+b5*b5+b6*b6+b7*b7+b8*b8+b9*b9+b10*b10+b11*b11)/11)',
    {
        b2: ks1.select('B2'), b3: ks1.select('B3'),b4: ks1.select('B4'), 
        b5: ks1.select('B5'), b6: ks1.select('B6'), b7: ks1.select('B7'),
        b8: ks1.select('B8'), b9: ks1.select('B8A'), b10: ks1.select('B11'), 
        b11: ks1.select('B12')
    }).rename('rmse');
    */
    var scaled = col.multiply(10000);
    var sma = ee.Image(scaled).set('system:time_start',img.get('system:time_start'));

    var soil = sma.select('SL').rename('Soil');
    var gvFraction = sma.select('GV').rename('GV');
    var daFraction = sma.select('DA').rename('DA');
    var npvFraction = ee.Image.constant(10000)
      .subtract(sma.select('SL'))
      .subtract(sma.select('GV'))
      .subtract(sma.select('DA'))
      .max(0)
      .rename('NPV');

    return soil
      .addBands([gvFraction, npvFraction, daFraction])
      .toInt()
      .set('system:time_start', img.get('system:time_start'));
  };
  var SMA = unmixed(image);
  return SMA;
};

//获取解混序列影像
//S_year开始年份//E_year结束年份//S_Day年内开始日期//E_Day年内结束日期
//mergeDay几天合并为1幅//AOI研究区
exports.Get_SMAseries_images = function (S_year,E_year,S_Day,E_Day,mergeDay,AOI){
  //resampling all Bands to 10m
  var s2_resample = function(image){
    var proj_10m = image.select('B2').projection();//meta crs
    var bands_10m = image.select(['B2','B3','B4','B8']);
    var bands_20m = image.select(['B5','B6','B7','B8A','B11','B12']);
        bands_20m = bands_20m.reproject(proj_10m,null,10).rename(['B5','B6','B7','B8A','B11','B12']);
    return bands_20m.addBands(bands_10m).select(['B2', 'B3','B4','B5','B6','B7','B8','B8A','B11','B12']);
  };
  
  var dataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(AOI)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',60))
    .filterDate(ee.Date.fromYMD(S_year,1,1),ee.Date.fromYMD(E_year,12,31))
    .filter(ee.Filter.dayOfYear(S_Day, E_Day))
    .map(s2_resample)
    .sort('system:time_start');
  
  var cloudProbabilityCollection = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
    .filterBounds(AOI);
    
  /*********************SMA**********************************/
  var s2_result_col = dataset.map(function(s2_result){
    var SMA_result1 = LSMA(s2_result);
    return ee.Image(SMA_result1);
  });
  
  function MergeFunc (ImageCollection){
    ImageCollection = ee.ImageCollection(ImageCollection);
    
    var five_Dates1 = ee.List.sequence(S_year,E_year,1).map(function(Year_Num){
      var Lyear = ee.List.sequence(S_Day,E_Day,mergeDay).map(function(Num){
        return ee.Date.fromYMD(Year_Num,1,1).advance(Num,'day').millis();
      }).flatten();
      return Lyear;
    }).flatten().sort();
    
    var Merge_ImageCollection1 = five_Dates1.map(function(date){
      var SS_date = ee.Date(date);
      var EE_date = SS_date.advance(mergeDay, 'day');
      
      var Images = ImageCollection.filter(ee.Filter.date(SS_date, EE_date));
      var s2Clouds = cloudProbabilityCollection.filter(ee.Filter.date(SS_date, EE_date));
      
      var s2SrWithCloudMask = ee.Join.saveFirst('cloud_mask').apply({
        primary: Images,
        secondary: s2Clouds,
        condition:
          ee.Filter.equals({leftField: 'system:index', rightField: 'system:index'})
      });
      
      var Num = s2SrWithCloudMask.size();
      var merge_image = s2SrWithCloudMask.median();
      merge_image = merge_image.set('system:time_start',SS_date.millis())
                               .set('ImageNumber',Num);
      return ee.Image(merge_image);
    });
    
    ImageCollection = ee.ImageCollection(Merge_ImageCollection1).sort('system:time_start');
    ImageCollection = ImageCollection.filter(ee.Filter.gt('ImageNumber',0));
    return ImageCollection;
  }
  
  function AdjustTime (ImageCollection){
    var DIF_UNIT = ee.Number(365).subtract(E_Day).add(S_Day).multiply(-1);
    ImageCollection = ee.ImageCollection(ImageCollection);
    var AdjustTimeIMGS = ImageCollection.map(function(IMG){
      var IMG_Date = ee.Date(IMG.date());
      var IMG_Year = ee.Number(IMG_Date.get('year'));
      var DIF_Year = IMG_Year.subtract(S_year);
      var DIF_INDX = DIF_Year.multiply(DIF_UNIT);
      var New_Date = IMG_Date.advance(DIF_INDX,'day');
      return ee.Image(IMG).set('system:time_start',New_Date.millis());
    });
    return ee.ImageCollection(AdjustTimeIMGS);
  }
  
  s2_result_col = MergeFunc(s2_result_col);
  var Out_Result = AdjustTime(s2_result_col);
  
  return ee.ImageCollection(Out_Result);
};

////////////////////////////////////////////////////////////
//谐波拟合
//year年份//S_date年内开始日期//E_date年内结束日期
//collection拟合的原始序列数据//order谐波阶数
exports.Harmonic_SMA_analysis = function (year,S_date,E_date,collection,order){
	/*************harmonic fitting********************/
	var linear_coefficients = [1,0];
	
	var har_GV = Harmonic_Fitting(collection,'GV',linear_coefficients,order,S_date,E_date);
	var har_DA = Harmonic_Fitting(collection,'DA',linear_coefficients,order,S_date,E_date);
	var har_Soil = Harmonic_Fitting(collection,'Soil',linear_coefficients,order,S_date,E_date);
	var har_NPV = Harmonic_Fitting(collection,'NPV',linear_coefficients,order,S_date,E_date);
	
	var har_GV2 = ee.Image(har_GV).select('coefficients').set({element:'GV',year:year}).rename('GV');
	var har_DA2 = ee.Image(har_DA).select('coefficients').set({element:'DA',year:year}).rename('DA');
	var har_Soil2 = ee.Image(har_Soil).select('coefficients').set({element:'Soil',year:year}).rename('Soil');
	var har_NPV2 = ee.Image(har_NPV).select('coefficients').set({element:'NPV',year:year}).rename('NPV');
	
	var har_GV3 = ee.Image(har_GV).select('RMSE').set({element:'GV',year:year}).rename('RMSE_GV');
	var har_DA3 = ee.Image(har_DA).select('RMSE').set({element:'DA',year:year}).rename('RMSE_DA');
	var har_Soil3 = ee.Image(har_Soil).select('RMSE').set({element:'Soil',year:year}).rename('RMSE_Soil');
	var har_NPV3 = ee.Image(har_NPV).select('RMSE').set({element:'NPV',year:year}).rename('RMSE_NPV');
	
	var HAR = ee.Image.cat([har_GV2,har_DA2,har_Soil2,har_NPV2]);
	var RMSE= ee.Image.cat([har_GV3,har_DA3,har_Soil3,har_NPV3]);
	
	return ee.Image.cat([HAR,RMSE]);
};

////////////////////////////////////////////////////////////
exports.Calculate_Vegetation_indexs = function (Metacollection){
  Metacollection = Metacollection.select(['B2','B3','B4','B8','B11','B12','B5'],['BLUE','GREEN','RED','NIR','SWIR1','SWIR2','RE1']);
  var VI_Cellection = Metacollection.map(function(image){
    var MAP = {BLUE:ee.Image(image.select('BLUE')),GREEN:ee.Image(image.select('GREEN')),RED:ee.Image(image.select('RED')),NIR:ee.Image(image.select('NIR')),
    SWIR1:ee.Image(image.select('SWIR1')),SWIR2:ee.Image(image.select('SWIR2')),RE1:ee.Image(image.select('RE1'))};
    
    var NDVI = image.expression('(NIR-RED)/(NIR+RED)',MAP).rename('NDVI');
    var EVI  = image.expression('2.5*(NIR-RED)/((NIR+RED*6-BLUE*7.5)+1)',MAP).rename('EVI');
    var EVI2 = image.expression('2.4*(NIR-RED)/(NIR+RED+1)',MAP).rename('EVI2');
    var MTCI = image.expression('(NIR-RE1)/(RE1-RED)',MAP).rename('MTCI');
    var GCVI = image.expression('(NIR/GREEN)-1',MAP).rename('GCVI');
    var TVI  = image.expression('60*(NIR-GREEN)-100*(RED-GREEN)',MAP).rename('TVI');
    var SAVI = image.expression('(NIR-RED)/(NIR+RED+0.5)*1.5',MAP).rename('SAVI');
    var NDWI = image.expression('(NIR-SWIR1)/(NIR+SWIR1)',MAP).rename('NDWI');
    var NDWI2= image.expression('(NIR-SWIR2)/(NIR+SWIR2)',MAP).rename('NDWI2');
    var MODCRC = image.expression('(SWIR1-GREEN)/(SWIR1+GREEN)',MAP).rename('MODCRC');
    var NDTI = image.expression('(SWIR1-SWIR2)/(SWIR1+SWIR2)',MAP).rename('NDTI');
    var STI  = image.expression('(SWIR1/SWIR2)',MAP).rename('STI');
    return ee.Image([NDVI,EVI,EVI2,MTCI,GCVI,TVI,SAVI,NDWI,NDWI2,MODCRC,NDTI,STI]).set('system:time_start',image.get('system:time_start'));
  });
  return VI_Cellection;
};

////////////////////////////////////////////////////////////
// singleband harmonic fitting
exports.Harmonic_analysis_singleband = function (year,S_date,E_date,collection,band,order){
/*************harmonic fitting********************/
var linear_coefficients = [1,0];
var har_band = Harmonic_Fitting(collection,band,linear_coefficients,order,S_date,E_date);
har_band = ee.Image(har_band).set({element:band,year:year});
return ee.Image(har_band);
};

/***********************************************************************************************************************/
/***********************************************************************************************************************/
/**************************************************Function*************************************************************/
/***********************************************************************************************************************/
/***********************************************************************************************************************/
  //时序拟合函数，形参：
//1.TimeSeries需拟合的时间序列，需包含system:time_start属性 ee.ImageCollection
//2.拟合的模型，详细定义见函数开头 int或string
//3.因变量 string
//4.线性拟合选项(optional) 1*6 list 需要拟合的项为1，否则为0
//5.谐波拟合选项(optional) 1*6 list

var Harmonic_Fitting = function (TimeSeries,dependent,linear_coefficients,order,S_date,E_date){
var independents_linear=ee.List(['a0','slope']);
var independents_harmonic=ee.List(['cos1t','sin1t','cos2t','sin2t','cos3t','sin3t','cos4t','sin4t','cos5t','sin5t','cos6t','sin6t',
'cos7t','sin7t','cos8t','sin8t','cos9t','sin9t','cos10t','sin10t','cos11t','sin11t','cos12t','sin12t']);
var HF_list = ee.List.repeat(1,order*2);
HF_list =  ee.Algorithms.If(ee.Number(order).lt(12), HF_list.cat(ee.List.repeat(0,(12-order)*2)), HF_list);

linear_coefficients = ee.Array(linear_coefficients);
var harmonic_coefficients = ee.Array(HF_list);
linear_coefficients=linear_coefficients.multiply(ee.Array([1,2])).subtract(1).toList().removeAll(ee.List([-1]));
harmonic_coefficients=harmonic_coefficients.multiply(ee.Array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])).subtract(1).toList().removeAll(ee.List([-1]));

var independents=linear_coefficients.map(function(i){
  return ee.String(independents_linear.get(ee.Number(i)));
}).cat(harmonic_coefficients.map(function(i){
  return ee.String(independents_harmonic.get(ee.Number(i)));
}));
var T = ee.Date(E_date).difference(ee.Date(S_date),'day');
var TimeSeries_addVariable = TimeSeries.map(function(img){
  var img_dependent=img.select(dependent);
  var date = img.get('system:time_start');
  var time = ee.Image(ee.Date(date).difference(S_date,'day'));

  var PI2 = 2.0 * Math.PI;
  var OMEGAS =  ee.Number(PI2).divide(T);
  var img_coefs = img_dependent
         .addBands(ee.Image.constant(1.0).rename('a0'))
         .addBands(time.rename('slope'))
         //.addBands((time.multiply(OMEGAS/2)).cos().rename('cost/2'))
         //.addBands((time.multiply(OMEGAS/2)).sin().rename('sint/2'))
         .addBands((time.multiply(OMEGAS.multiply(1))).cos().rename('cos1t'))
         .addBands((time.multiply(OMEGAS.multiply(1))).sin().rename('sin1t'))
         .addBands((time.multiply(OMEGAS.multiply(2))).cos().rename('cos2t'))
         .addBands((time.multiply(OMEGAS.multiply(2))).sin().rename('sin2t'))
         .addBands((time.multiply(OMEGAS.multiply(3))).cos().rename('cos3t'))
         .addBands((time.multiply(OMEGAS.multiply(3))).sin().rename('sin3t'))
         .addBands((time.multiply(OMEGAS.multiply(4))).cos().rename('cos4t'))
         .addBands((time.multiply(OMEGAS.multiply(4))).sin().rename('sin4t'))
         .addBands((time.multiply(OMEGAS.multiply(5))).cos().rename('cos5t'))
         .addBands((time.multiply(OMEGAS.multiply(5))).sin().rename('sin5t'))
         .addBands((time.multiply(OMEGAS.multiply(6))).cos().rename('cos6t'))
         .addBands((time.multiply(OMEGAS.multiply(6))).sin().rename('sin6t'))
         .addBands((time.multiply(OMEGAS.multiply(7))).cos().rename('cos7t'))
         .addBands((time.multiply(OMEGAS.multiply(7))).sin().rename('sin7t'))
         .addBands((time.multiply(OMEGAS.multiply(8))).cos().rename('cos8t'))
         .addBands((time.multiply(OMEGAS.multiply(8))).sin().rename('sin8t'))
         .addBands((time.multiply(OMEGAS.multiply(9))).cos().rename('cos9t'))
         .addBands((time.multiply(OMEGAS.multiply(9))).sin().rename('sin9t'))
         .addBands((time.multiply(OMEGAS.multiply(10))).cos().rename('cos10t'))
         .addBands((time.multiply(OMEGAS.multiply(10))).sin().rename('sin10t'))
         .addBands((time.multiply(OMEGAS.multiply(11))).cos().rename('cos11t'))
         .addBands((time.multiply(OMEGAS.multiply(11))).sin().rename('sin11t'))
         .addBands((time.multiply(OMEGAS.multiply(12))).cos().rename('cos12t'))
         .addBands((time.multiply(OMEGAS.multiply(12))).sin().rename('sin12t'))
         .double();
  return img_coefs.set('system:time_start',date);
});
TimeSeries_addVariable = TimeSeries_addVariable.select(independents.add(dependent));
var trend = TimeSeries_addVariable
            .reduce(ee.Reducer.linearRegression({numX:linear_coefficients.cat(harmonic_coefficients).length(),numY: 1}));
var coefficients = trend.select('coefficients').arrayProject([0]).arrayFlatten([independents]);
var residuals =  trend.select('residuals').arrayProject([0]).arrayFlatten([['RMSE']]).rename('RMSE');
//return ee.Image(residuals).addBands(coefficients); //返回拟合序列、拟合残差和系数
return coefficients.addBands(residuals);
}

function addNumberID (IMC){
  IMC = IMC.sort('system:time_start')
  IMC = IMC.toList(IMC.size());
  var dataset_AddID = ee.List.sequence(0,IMC.size().subtract(1),1).map(function(X){
  var img = IMC.get(X);
  return ee.Image(img).set('Num',X);
  });
  return ee.ImageCollection(dataset_AddID).sort('system:time_start');
} 

function SG_filtering (IMC){
  var oeel=require('users/OEEL/lib:loadAll');
  IMC = ee.ImageCollection(IMC);
  IMC = addNumberID(IMC);
  var SG = oeel.ImageCollection.SavatskyGolayFilter(IMC,
  ee.Filter.maxDifference(5, 'Num', null, 'Num'),
  function(infromedImage,estimationImage){
        return ee.Image.constant(ee.Number(infromedImage.get('system:time_start'))
          .subtract(ee.Number(estimationImage.get('system:time_start'))));},
  3,['NPV','GV','DA','Soil']).select(['d_0_NPV','d_0_GV','d_0_DA','d_0_Soil'],['NPV','GV','DA','Soil']);
  return ee.ImageCollection(SG);
}

function addIndex (img){
  img = ee.Image(img);
  var GV = img.select('GV');
  var NPV = img.select('NPV');
  var Soil = img.select('Soil');
  var DA = img.select('DA');
  var GV_shade = GV.divide(ee.Image(10000).subtract(DA)).multiply(10000);
  var a1 = GV_shade.subtract(NPV).subtract(Soil);
  var a2 = GV_shade.add(NPV).add(Soil);
  var ndfi = ee.Image(a1.divide(a2)).multiply(10000).add(10000).rename('NDFI');
  //var b1 = Soil.subtract(NPV);
  //var b2 = Soil.add(NPV);
  //var ndoi = ee.Image(b1.divide(b2)).multiply(10000).add(10000).rename('NDOI');
  return img.addBands(ndfi);//.addBands(ndoi);
}

function remove_abnormal_values(images){
  var clean_images = images.map(function(image){
    var gv = image.select('GV').max(0);  // Set negative values to 0
    var npv = image.select('NPV').max(0);
    var soil = image.select('Soil').max(0);
    var da = image.select('DA').max(0);
    return ee.Image.cat([gv, npv, soil, da])
             .copyProperties(image, ['system:index', 'system:time_start']);
  });
  return ee.ImageCollection(clean_images);
}

exports.fit_phase_amplitude = function (img, T){
    img = ee.Image(img);
    var sin = img.select('.*sin.*');
    var cos = img.select('.*cos.*');
    var phase = sin.atan2(cos)
      // Scale to [0, 1] from radians. 
      .unitScale(-3.14159265359, 3.14159265359)
      .multiply(T); // To get phase in days!
    var amplitude = sin.hypot(cos);
    var phaseNames = phase.bandNames().map(function(x){return ee.String(x).replace('sin', 'phase')});
    var amplitudeNames = amplitude.bandNames().map(function(x){return ee.String(x).replace('sin', 'amplitude')});
    return phase.rename(phaseNames).addBands(amplitude.rename(amplitudeNames));
}
