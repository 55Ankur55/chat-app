const mongoose = require('mongoose');
const Schema = mongoose.Schema;

// Create Schema for Users
const UserSchema = new Schema({
    name: {
        type: String,
        required: true,
    },
    username: {
        type: String,
        required: true,
    },
    password: {
        type: String,
        required: true,
    },
    date: {
        type: String,
        default: Date.now,
    },
    agent: {
        type: Boolean,
        default: true,
    },
    available: {
        type: Boolean,
        default: false,
    },
});

module.exports = User = mongoose.model('users', UserSchema);
