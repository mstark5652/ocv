const request = require('request')

function requestImageRec (callback) {
  request('https://faces.azurewebsites.net/persons/identifyfromdocument/mix_eb543', function (error, response, body) {
    if (error) {
      return callback(null, error)
    }
    try {
      var json = JSON.parse(body);
      var splitName = json.name.split(' ');
      var first = splitName[0];
      callback(first)
    } catch (err) {
      callback(null, err)
    }
  })
}

requestImageRec(function (first, err) {
  if (err) {
    console.error('Error::', err)
  } else {
    console.log('First', first)
  }
})