  window.console = window.console || function(t) {};

  if (document.location.search.match(/type=embed/gi)) {
    window.parent.postMessage('resize', '*');
  }
const music = document.getElementById('music');

/*music.addEventListener('timeupdate', function() {
  getTime();
});*/
ChartIt(); 

async function ChartIt(){
  const data_prob =await getData_probabilite();
  const data_classe=await getData_classe();
  const data_ema=await getData_ema();


  const ctx_prob = document.getElementById('chart_probabilite');
  const ctx_classe = document.getElementById('chart_classe');
  const ctx_ema = document.getElementById('chart_ema');



  var dataset =[];
  var probabilite =[];
  var _data={};
  var  i,j ;
  var label_classe;
  
  for (i = 1; i < data_prob.classe.length; i++) {  
    label_classe = data_prob.classe[i];
    for (j=0 ;j<data_prob.y_classe.length ;j++)  {
      
      probabilite.push(data_prob.y_classe[j][i-1]);
      //console.log(label , data.y_classe[j][i])
      
    } 
    var randomColor = '#'+Math.floor(Math.random()*16777215).toString(16);
    var dataset_classe = {
                label:label_classe,
                data: probabilite,
                fill : false ,
                borderColor:  randomColor ,
                borderWidth: 2,
                pointHoverBackgroundColor : 'black'

            };

    dataset.push(dataset_classe);
    probabilite=[]
  }
  //console.log(dataset)
  
  _data = {
    labels: data_prob.xlables,
    datasets: dataset
  };


  const myChart = new Chart(ctx_prob, {

        type: 'line',
        data: _data,
        options: {
            scales: {
                yAxes: [{
                    ticks: {
                        beginAtZero: true
                    }
                }]
            },
          plugins: {
                  zoom: {
                    pan: {
                      enabled: true,
                      mode: 'xy',
                      rangeMin: {
                        x: null,
                        y: 0
                      },
                      rangeMax: {
                        x: null,
                        y: 1
                      },
                      speed: 30,
                      threshold: 10,
                      onPan: function({chart}) {},
                      onPanComplete: function({chart}) { }
                    },
                    zoom: {
                      enabled: true,
                      drag: true,
                      mode: 'xy',
                      rangeMin: {
                        x: null,
                        y: 0
                      },
                      rangeMax: {
                        x: null,
                        y: 1
                      },
                      speed: 0.1,
                      threshold: 2,
                      sensitivity: 5,
                      onZoom: function({chart}) { },
                      onZoomComplete: function({chart}) {  }
                    }
                  }
                },
                scales: {
                      yAxes: [{
                        scaleLabel: {
                           display: true,
                           labelString: 'probabilité'
                        }
                     }],
                     xAxes: [{
                        scaleLabel: {
                           display: true,
                           labelString: 'Temps(ms)'
                        }
                     }]
        }

        }
    });
  window.resetZoom = function() {
            myChart.resetZoom();
        };
  var ylabels ={};
  for (i = 1; i < data_prob.classe.length; i++) {  
    ylabels[i] = data_prob.classe[i];
  }
  var __data =[];
  for (i = 0; i < data_classe.y_pred.length; i++) {  

    for (j=1 ;j<data_prob.classe.length ;j++)  {
      if(data_prob.classe[j]== data_classe.y_pred[i]){
        __data.push(j);
      }
           
    }
  } 
    
  const myChart_1= new Chart(ctx_classe, {
        type: 'line',
        data: {
              labels: data_classe.xlables,
              datasets:[{
                label:'La classe prévue pour chaque unité de temps',
                data:__data ,
                fill : false ,
                borderColor:  'rgb(0, 99, 132)',
                borderWidth: 1,
                pointHoverBackgroundColor : 'black'

        }]
        },
        options: {
            scales: {
                yAxes: [{
                  scaleLabel: {
                           display: true,
                           labelString: 'Classes'
                  },
                  ticks: {
                    max: data_prob.classe.length-1,
                    min: 1,
                    stepSize:1,
                    callback: function(value, index, values) {
                    return ylabels[value] + " -" +value +"-"  ;
                }
                  }

                    
                }],
                xAxes: [{

                      scaleLabel: {
                       display: true,
                        labelString: 'Temps(ms)'

                      }
                 }]
            }
             
        }
    });


var ema = [];
dataset=[];
//console.log( data_ema.y_classe);
for (i = 1; i < data_ema.classe.length; i++) {  
    label_classe = data_ema.classe[i];
    for (j=0 ;j<data_ema.y_classe.length ;j++)  {
      
      ema.push(data_ema.y_classe[j][i-1]);
      //console.log(label , data.y_classe[j][i])
      
    } 
    
    var randomColor = '#'+Math.floor(Math.random()*16777215).toString(16);
    var dataset_ema = {
                label:label_classe+"_EMA",
                data: ema,
                fill : false ,
                borderColor:  randomColor ,
                borderWidth: 1
            };

    dataset.push(dataset_ema);
    ema=[]
  }
  //console.log(dataset)
  
  __data = {
    labels: data_ema.xlables,
    datasets: dataset
  };
 // console.log( __data);


 const myChart_2 = new Chart(ctx_ema, {
        type: 'line',
        data: __data,
        options: {
            scales: {
                yAxes: [{
                    ticks: {
                        beginAtZero: true
                    }
                }]
            },
           plugins: {
                  zoom: {
                    pan: {
                      enabled: true,
                      mode: 'xy',
                      rangeMin: {
                        x: null,
                        y: null
                      },
                      rangeMax: {
                        x: null,
                        y: 1
                      },
                      speed: 30,
                      threshold: 10,
                      onPan: function({chart}) {  },
                      onPanComplete: function({chart}) { }
                    },
                    zoom: {
                      enabled: true,
                      drag: true,
                      mode: 'xy',
                      rangeMin: {
                        x: null,
                        y: 0
                      },
                      rangeMax: {
                        x: null,
                        y: 1
                      },
                      speed: 0.1,
                      threshold: 2,
                      sensitivity: 5,
                      onZoom: function({chart}) { },
                      onZoomComplete: function({chart}) {  }
                    }
                  }
                },
                scales: {
                      yAxes: [{
                        scaleLabel: {
                           display: true,
                           labelString: 'probabilité'
                        }
                     }],
                     xAxes: [{
                        scaleLabel: {
                           display: true,
                           labelString: 'Temps(ms)'
                        }
                     }]
        }

        }
    });
window.resetZoom_2 = function() {
            myChart_2.resetZoom();
        };
  }


//Fonctions de récupération des données
  async function getData_probabilite(){
    const xlables = [];
    const y_classe=[];
    var cl = [];

	  const response = await fetch('https://raw.githubusercontent.com/micbuffa/stageAyaMachineLearningGuitare/master/src/Test/Test-6-JV%26GR-Presets/Probabilite_classe_predictions.csv');
    var data = await response.text();
    data = data.trim();

    var table = data.split('\n');

    const classe =table[0].split(',');

    const nb_classes = classe.length;
    console.log(nb_classes)
    
    table = data.split('\n').slice(1);

    table.forEach(row => {
        const col = row.split(',');
        const temps = col[0];
        xlables.push(temps);

        var i ;
        cl =[];
        for (i = 1; i < nb_classes; i++) {
          cl.push(col[i]);
        }
        y_classe.push(cl);

    });
    console.log(y_classe);
    
    return {xlables,y_classe,classe}
  }

  async function getData_classe(){
    const xlables = [];
    const y_pred=[];

	  const response = await fetch('https://raw.githubusercontent.com/micbuffa/stageAyaMachineLearningGuitare/master/src/Test/Test-6-JV%26GR-Presets/Classes_predictions.csv');
    var data = await response.text();
    data = data.trim();
    const table = data.split('\n').slice(1);
 
    table.forEach(row => {
        const col = row.split(',');
        const temps = col[0];
        const classe = col[1]
        xlables.push(temps);
        y_pred.push(classe);


    });
    //console.log(xlables,y_pred);
    
    return {xlables,y_pred}
  }

  async function getData_ema(){
    const xlables = [];
    const y_classe=[];
    var cl = [];

	  const response = await fetch('https://raw.githubusercontent.com/micbuffa/stageAyaMachineLearningGuitare/master/src/Test/Test-6-JV%26GR-Presets/EMA_predictions.csv');
    var data = await response.text();
    data = data.trim();
    var table = data.split('\n');
    const classe =table[0].split(',');
    const nb_classes = classe.length;
    //console.log(classe[0]);
    table = data.split('\n').slice(1);
    table.forEach(row => {
        const col = row.split(',');
        const temps = col[0];
        xlables.push(temps);

        var i ;
        cl =[];
        for (i = 1; i < nb_classes; i++) {
          
          cl.push(col[i]);

        }
        y_classe.push(cl);


    });
    //console.log(y_classe,classe);
    
    return {xlables,y_classe,classe}
  }
 
  /*async function getTime(){

    console.log(Math.floor(music.currentTime*1000));
    
    

  }*/
'use strict';
var wavesurfer;
// Init & load audio file
document.addEventListener('DOMContentLoaded', function() {
    // Init
    wavesurfer = WaveSurfer.create({
        container: document.querySelector('#waveform'),
        waveColor: '#A8DBA8',
        progressColor: '#3B8686',
        backend: 'MediaElement',
        mediaControls: true
    });
wavesurfer.load('https://github.com/micbuffa/stageAyaMachineLearningGuitare/blob/master/src/Test/Test-6-JV%26GR-Presets/Test-6-JV%26GR-Presets.wav?raw=true');
});