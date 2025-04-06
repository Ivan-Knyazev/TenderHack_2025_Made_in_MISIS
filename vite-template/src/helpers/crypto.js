const bcrypt = require('bcrypt');
const fixedSalt = '$2b$10$abcdefghijklmnopqrstuv';

export function decode(text) {
    return bcrypt.hashSync(text, fixedSalt);
}

