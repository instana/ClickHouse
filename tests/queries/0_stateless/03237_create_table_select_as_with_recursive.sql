drop table if exists t;
create view t AS (WITH RECURSIVE 42 as ttt SELECT ttt) -- { clientError SYNTAX_ERROR };
