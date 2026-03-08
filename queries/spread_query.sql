WITH puts AS ( 
    SELECT
        c.*,
        (c.bid + c.ask) / 2.0 AS mid,
        e.exp_date,
        s.snap_date,
        e.id   AS exp_id,
        s.id   AS snap_id,
        y.sym  AS symbol,
        (c.ask - c.bid) / ((c.ask + c.bid) / 2.0) AS spread
    FROM   contracts   c
    JOIN   expirations  e  ON e.id = c.expiration_id
    JOIN   snapshots    s  ON s.id = e.snapshot_id
    JOIN   symbols      y  ON y.id = s.symbol_id
    WHERE  c.opt_type = 'P'
      AND  julianday(e.exp_date) - julianday(s.snap_date) BETWEEN 30 AND 90
      AND  c.bid   > 0
      AND  c.ask   > 0
      AND  (c.ask - c.bid) / ((c.ask + c.bid) / 2.0) <= 0.05
      AND  c.volume        > 50
      AND  c.open_interest > 100
)

SELECT
    short.symbol                 AS symbol,
    short.snap_date              AS snap_date,
    short.exp_date               AS expiry,
    short.strike                 AS short_strike,
    long.strike                  AS long_strike,

    -- use mids
    short.mid - long.mid         AS credit,
    short.strike - long.strike   AS width,
    (short.mid - long.mid) /
    (short.strike - long.strike) AS cpw,
    short.delta                  AS short_delta,

    short.iv                     AS short_iv,

    -- edge (mid-based)
    ((short.mid - long.mid) - ABS(short.delta) * (short.strike - long.strike))
      / ((short.strike - long.strike) - (short.mid - long.mid)) AS edge,

    -- max loss per spread (mid-based)
    (short.strike - long.strike) - (short.mid - long.mid)       AS risk,

    -- EV  =  credit  - |Δ| × width (mid-based)
    (short.mid - long.mid) - ABS(short.delta) * (short.strike - long.strike) AS ev,

    short.open_interest          AS short_OI,
    short.volume                 AS short_volume,

    short.spread as short_spread,
    long.spread as long_spread

FROM   puts  AS short
JOIN   puts  AS long
       ON    long.exp_id   = short.exp_id
      AND    long.snap_id  = short.snap_id
      AND    long.symbol   = short.symbol
      AND    long.strike   < short.strike

WHERE  (short.mid - long.mid) /
       (short.strike - long.strike) > ABS(short.delta)          -- x > Δ
  AND  ABS(short.delta) BETWEEN 0.10 AND 0.25                   -- 10–25 Δ short leg
  AND  ((short.mid - long.mid) - ABS(short.delta) * (short.strike - long.strike))
       / ((short.strike - long.strike) - (short.mid - long.mid)) > 0.03  -- edge > 3%

ORDER  BY edge DESC;
