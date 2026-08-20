// Serves this directory over HTTPS with a local self-signed certificate.
//
// DeviceMotionEvent is only exposed by modern browsers in a secure context, so
// plain HTTP (python -m http.server) is not sufficient for the phone demo --
// see the "Phone Streaming Demo" section of the README.
//
// Usage (from the project root, after generating .local_certs/ -- see README):
//     node phone_client/serve_https.js

const fs = require('fs');
const path = require('path');
const https = require('https');

const PORT = 8080;
const ROOT = __dirname;
const CERT_DIR = path.join(__dirname, '..', '.local_certs');

const MIME_TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };

const options = {
  key: fs.readFileSync(path.join(CERT_DIR, 'key.pem')),
  cert: fs.readFileSync(path.join(CERT_DIR, 'cert.pem')),
};

https.createServer(options, (req, res) => {
  let filePath = path.join(ROOT, req.url === '/' ? 'index.html' : req.url);
  filePath = path.normalize(filePath);
  if (!filePath.startsWith(ROOT)) { res.writeHead(403); res.end('Forbidden'); return; }

  fs.readFile(filePath, (err, data) => {
    if (err) { res.writeHead(404); res.end('Not found'); return; }
    const ext = path.extname(filePath);
    res.writeHead(200, { 'Content-Type': MIME_TYPES[ext] || 'application/octet-stream' });
    res.end(data);
  });
}).listen(PORT, '0.0.0.0', () => {
  console.log(`Serving phone_client/ over HTTPS on port ${PORT}`);
});
