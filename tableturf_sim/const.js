const fs = require('fs');
const path = require('path');
const { createCanvas } = require('canvas');

const CARD_PATH = 'data/cards/MiniGameCardInfo.json';
const OUTPUT_DIR = 'data/cards/images';

if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

const cards = JSON.parse(fs.readFileSync(CARD_PATH, 'utf8'));

const GRID_SIZE = 8;          // 8×8
const CELL_SIZE = 16;         // 每格 16px
const CANVAS_SIZE = GRID_SIZE * CELL_SIZE;

const COLOR_MAP = {
    Empty: '#000000',   // 黑色
    Fill: '#ff8c00',    // 橙色
    Special: '#ffd700'  // 黄色
};

for (const card of cards) {
    if (!card.Square || card.Square.length !== 64) {
        console.warn(`Skip card ${card.__RowId}, invalid Square`);
        continue;
    }

    const canvas = createCanvas(CANVAS_SIZE, CANVAS_SIZE);
    const ctx = canvas.getContext('2d');

    // 背景
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    card.Square.forEach((cell, i) => {
        const row = Math.floor(i / GRID_SIZE);
        const col = i % GRID_SIZE;

        ctx.fillStyle = COLOR_MAP[cell] ?? '#ff00ff'; // 未知类型紫色报警
        ctx.fillRect(
            col * CELL_SIZE,
            row * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE
        );
    });

    const filename = `${card.__RowId}.png`;
    const outPath = path.join(OUTPUT_DIR, filename);

    fs.writeFileSync(outPath, canvas.toBuffer('image/png'));
    console.log(`✔ Generated ${outPath}`);
}
