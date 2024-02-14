$(document).ready(function() {
    // Initialize the dropdowns
    $('#dataset-select').multiselect({
        nonSelectedText: 'Select Dataset',
        enableFiltering: true,
        buttonWidth: '100%'
    });
    $('#length-select').multiselect({
        nonSelectedText: 'Select Length',
        enableFiltering: true,
        buttonWidth: '100%'
    });
});

var modal = document.getElementById("myModal");

// Function to show modal
function showModal() {
    modal.style.display = "block";
}

// Function to hide modal
function hideModal() {
    modal.style.display = "none";
}

async function index_button_click() {
    // let storedData = JSON.parse(localStorage.getItem('myData'));
    let dataset = $('#dataset-select').val();
    let length = $('#length-select').val();
    let index = document.getElementById("index").value;
    console.log(index)
    // alert("按钮被点击!");
    var url = '/forecast?datasets=' + dataset + '&lengths=' + length + '&index=' + index;
    showModal();
    // Fetch data
    let response = await fetch(url);
    let data = await response.json();
    // localStorage.setItem('my_dataloader', JSON.stringify(data.data_loader));    

    console.log("return data: ", data)

    // Update the plot
    console.log("update the plot")
    var trace1 = {
        x: data.time,
        y: data.values,
        mode: 'lines',
        name: 'Look back window',
        line: {
            color: 'rgb(55, 128, 191)',
            width: 2
        }
    };
    console.log(trace1)

    var trace2 = {
        x: data.prediction_time,
        y: data.prediction,     
        mode: 'lines',
        name: 'Prediction',
        line: {
            color: 'rgb(219, 64, 82)',
            width: 2
        }
    };
    console.log(trace2)
    
    var layout = {
        title: 'Prediction Data',
        xaxis: {
            title: 'Time'
        },
        yaxis: {
            title: 'Value'
        }
    };
    
    var plotData = [trace1, trace2];
    
    Plotly.newPlot('plotly-chart', plotData, layout);

    // Update the table
    let tableBody = document.querySelector('table tbody');
    tableBody.innerHTML = '';  // Clear any existing rows
    
    // Assume each inner array in data.values and data.time is of the same length
    for (let i = 0; i < data.values.length; i++) {
        let row = tableBody.insertRow();
        let cell0 = row.insertCell(0);
        let cell1 = row.insertCell(1);
        let cell2 = row.insertCell(2);
        cell0.textContent = i + 1;  // Row number
        cell1.textContent = data.time[i];
        cell2.textContent = data.values[i];
    }
    hideModal();
}

// Function to fetch forecast data and update the table and plot
async function fetchForecastData() {
    try {
        // Get selected datasets and predict lengths
        let dataset = $('#dataset-select').val();
        let length = $('#length-select').val();
        // let index = $('#slider').val();
        
        // Form the query string
        var url = '/forecast?datasets=' + dataset + '&lengths=' + length;

        // Fetch data
        showModal();
        let response = await fetch(url);
        let data = await response.json();
        // localStorage.setItem('my_dataloader', JSON.stringify(data.data_loader));    

        console.log("return data: ", data)

        // Update the plot
        console.log("update the plot")
        var trace1 = {
            x: data.time,
            y: data.values,
            mode: 'lines',
            name: 'Look back window',
            line: {
                color: 'rgb(55, 128, 191)',
                width: 2
            }
        };
        console.log(trace1)

        var trace2 = {
            x: data.prediction_time,
            y: data.prediction,
            mode: 'lines',
            name: 'Prediction',
            line: {
                color: 'rgb(219, 64, 82)',
                width: 2
            }
        };
        console.log(trace2)
        
        var layout = {
            title: 'Prediction Data',
            xaxis: {
                title: 'Time'
            },
            yaxis: {
                title: 'Value'
            }
        };
        
        var plotData = [trace1, trace2];
        
        Plotly.newPlot('plotly-chart', plotData, layout);

        // Update the table
        let tableBody = document.querySelector('table tbody');
        tableBody.innerHTML = '';  // Clear any existing rows
        
        // Assume each inner array in data.values and data.time is of the same length
        for (let i = 0; i < data.values.length; i++) {
            let row = tableBody.insertRow();
            let cell0 = row.insertCell(0);
            let cell1 = row.insertCell(1);
            let cell2 = row.insertCell(2);
            cell0.textContent = i + 1;  // Row number
            cell1.textContent = data.time[i];
            cell2.textContent = data.values[i];
        }

        var container = document.getElementById("index_container");
        container.innerHTML = '';

        var label = document.createElement('label');
        label.setAttribute('for', 'index'); 
        label.textContent = 'Please select sample index: ';

        var input = document.createElement("input");
        input.type = "number";
        input.id = "index";
        input.max = data.data_max; 
        input.placeholder = "Max: "+(data.data_max-1); 


        var button = document.createElement("button");
        button.innerHTML = "predict";

        button.addEventListener("click", index_button_click);


        container.appendChild(label);
        container.appendChild(input);
        container.appendChild(button);

        hideModal();

    } catch (error) {
        console.error('Error:', error);
    }
}
