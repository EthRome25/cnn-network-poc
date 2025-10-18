const axios = require('axios');

async function testRPC() {
  try {
    const response = await axios.post('http://localhost:2428/jsonrpc', {
      jsonrpc: '2.0',
      id: 1,
      method: 'execute',
      params: {
        contextId: '4bZJB5vmPAPn7yYwPDwDfJmpEbTKdFm3pvcmSpZMDuYx',
        method: 'get_current_model',
        argsJson: {},
        executorPublicKey: ''
      }
    }, {
      headers: {
        'Content-Type': 'application/json'
      },
      timeout: 5000
    });
    
    console.log('Response:', JSON.stringify(response.data, null, 2));
  } catch (error) {
    if (error.response) {
      console.error('Error response:', error.response.status, error.response.data);
    } else if (error.request) {
      console.error('No response received:', error.message);
    } else {
      console.error('Error:', error.message);
    }
  }
}

testRPC();

