-- Q1. Customers with completed orders above INR 500
-- and total qualifying spend above INR 8000.
SELECT
    customer_id,
    SUM(order_amount) AS qualifying_spend
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY qualifying_spend DESC;


-- Q2. Revenue by city and product category.
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders AS o
JOIN customers AS c
  ON c.customer_id = o.customer_id
JOIN products AS p
  ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;


-- Q3. Products priced above the average catalogue price.
SELECT
    product_id,
    product_name,
    price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;


-- Q4. Customers with no completed order.
SELECT
    c.customer_id,
    c.city
FROM customers AS c
LEFT JOIN orders AS o
  ON o.customer_id = c.customer_id
 AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;


-- Q5. Customers whose completed spend exceeds
-- the average completed spend per customer.
WITH customer_spend AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
)
SELECT
    customer_id,
    total_spend
FROM customer_spend
WHERE total_spend > (SELECT AVG(total_spend) FROM customer_spend)
ORDER BY total_spend DESC;


-- Q6. Rank customers by completed spend within each city.
SELECT
    c.city,
    o.customer_id,
    SUM(o.order_amount) AS total_spend,
    RANK() OVER (
        PARTITION BY c.city
        ORDER BY SUM(o.order_amount) DESC
    ) AS spend_rank,
    DENSE_RANK() OVER (
        PARTITION BY c.city
        ORDER BY SUM(o.order_amount) DESC
    ) AS dense_spend_rank,
    ROW_NUMBER() OVER (
        PARTITION BY c.city
        ORDER BY SUM(o.order_amount) DESC, o.customer_id
    ) AS spend_row_number
FROM orders AS o
JOIN customers AS c
  ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
GROUP BY c.city, o.customer_id
ORDER BY c.city, spend_rank, o.customer_id;


-- Q7. Daily completed revenue and cumulative revenue.
WITH daily_revenue AS (
    SELECT
        substr(order_ts, 1, 10) AS day,
        SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY substr(order_ts, 1, 10)
)
SELECT
    day,
    ROUND(revenue, 2) AS daily_revenue,
    ROUND(
        SUM(revenue) OVER (ORDER BY day),
        2
    ) AS cumulative_revenue
FROM daily_revenue
ORDER BY day;
