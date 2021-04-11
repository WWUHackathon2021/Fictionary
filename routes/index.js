const express = require('express');
require('dotenv').config();
var router = express.Router();

/* GET home page. */
router.get('/', function (req, res, next) {
    res.render('index', {
        title: 'Fictionary',
        layout: false,
    });
});

module.exports = router;

/* Get image from text */
const http = require("https");

function convertTextToImage(str) {
    const text = encodeURIComponent(str)
    const color = "000000"
    const backgroundColor = "FFFFFF"
    const fontFamily = "verdana"
    const fontSize = "12"
    const imgType = "png"

    const params = `?text=${text}&fcolor=${color}&bcolor=${backgroundColor}&font=${fontFamily}&size=${fontSize}&type=${imgType}`;

    const options = {
        "method": "GET",
        "hostname": "img4me.p.rapidapi.com",
        "port": null,
        "path": params,
        "headers": {
            "x-rapidapi-key": "d541200d34msh8dbf8af02468f86p1ce650jsn3634ce9e5364",
            "x-rapidapi-host": "img4me.p.rapidapi.com",
            "useQueryString": true
        }
    };

    const req = http.request(options, function (res) {
        const chunks = [];

        res.on("data", function (chunk) {
            chunks.push(chunk);
        });

        res.on("end", function () {
            const body = Buffer.concat(chunks);
            console.log(body.toString());
        });
    });

    req.end();
}

convertTextToImage("Logan = a kind of lovable");

// ATTEMPTED TO USE FETCH API BUT THE RESPONSE WAS JUST THE REQUEST
// const fetch = require("node-fetch");

// async function convertTextToImage() {  
//     const text = encodeURIComponent("Logan = a kind of lovable")
//     const color = "000000"
//     const backgroundColor = "FFFFFF"
//     const fontFamily = "verdana"
//     const fontSize = "12"
//     const imgType = "png"

//     const params = `?text=${text}&fcolor=${color}&bcolor=${backgroundColor}&font=${fontFamily}&size=${fontSize}&type=${imgType}`;

//     const endpoint = `https://img4me.p.rapidapi.com/${params}`

//     const options = {
//         method: "GET",
//         mode: 'cors',
//         redirect: 'follow',
//         headers: {
//             "x-rapidapi-key": "d541200d34msh8dbf8af02468f86p1ce650jsn3634ce9e5364",
//             "x-rapidapi-host": "img4me.p.rapidapi.com",
//             "useQueryString": true
//         },
//     };

//     try {
//       const response = await fetch(endpoint, options);
//       if (response.ok) {
//         const jsonResponse = await response;
//         console.log(jsonResponse);
//       }
//     } catch (error) {
//       console.log(error);
//     }
// }

// convertTextToImage();
