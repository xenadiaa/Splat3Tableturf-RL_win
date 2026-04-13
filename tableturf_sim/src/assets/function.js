function compareSingleBoxFinalResult(player, enemy) {
    const epsilon = 0.000001; // 对应原代码中的 Epision

    const playerPriority = 30.0 + player.isUsingSP - player.cardCount + player.isSpBox;
    const enemyPriority = 30.0 + enemy.isUsingSP + enemy.cardCount + player.isSpBox;

    if (Math.abs(playerPriority - enemyPriority) < epsilon) {
        return "BT_Conflict";
    } else if (playerPriority > enemyPriority) {
        return player.boxType;
    } else {
        return enemy.boxType;
    }
}
