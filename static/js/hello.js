function init(){
	 $("#response").html("Welcome")
}

function predict(){
	$.post("/neuralnetwork/predict", {"info":$("#inp_pred").val()}, function(data, status){
				  $("#response").html(data)
			  }); 
}