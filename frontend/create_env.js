const fs = require('fs');

const envContent = 'NEXT_PUBLIC_API_URL=http://localhost:8000\n';

fs.writeFileSync('.env.local', envContent, 'utf-8');

console.log('.env.local file created successfully');
